"""Ensure the zhisa training instance is running, retrying on capacity.

Automation for the common AWS failure mode on scarce GPU instance types
(g6/g5/g4dn): ``StartInstances`` returns ``InsufficientInstanceCapacity``.
This script:

1. attempts ``StartInstances`` and, on insufficient capacity, sleeps and retries
   with a slow backoff (capacity slots free up unpredictably);
2. optionally allocates + associates an Elastic IP and opens SSH on the
   instance's security group (``--ensure-ssh``), making the box reachable;
3. waits until the instance reaches ``running`` and the status check passes;
4. prints reachability facts (public IP, SSH command) for the caller.

Uses the standard boto3 credential chain (default profile, region).
"""
from __future__ import annotations

import argparse
import sys
import time

import boto3

DEFAULT_INSTANCE = "i-016771ecc77ee6bf6"
DEFAULT_REGION = "eu-north-1"


def _ec2(region: str, profile: str | None):
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("ec2", region_name=region)


def _state(ec2, instance_id: str) -> str:
    try:
        r = ec2.describe_instance_status(
            InstanceIds=[instance_id], IncludeAllInstances=True
        )
        if r["InstanceStatuses"]:
            return r["InstanceStatuses"][0]["InstanceState"]["Name"]
    except Exception:
        pass
    r = ec2.describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0]["State"]["Name"]


def start_with_retry(
    ec2,
    instance_id: str,
    *,
    max_attempts: int = 120,
    base_backoff: float = 30.0,
) -> None:
    attempts = 0
    while True:
        attempts += 1
        try:
            ec2.start_instances(InstanceIds=[instance_id])
            print(f"[{attempts}] start-instances accepted")
            return
        except ec2.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            msg = exc.response.get("Error", {}).get("Message", str(exc))
            if "InsufficientInstanceCapacity" in (code + msg):
                if attempts >= max_attempts:
                    print(f"[{attempts}] giving up after {max_attempts} attempts (capacity).")
                    sys.exit(2)
                delay = base_backoff * (1.0 + (attempts % 5))
                print(
                    f"[{attempts}] insufficient capacity; retrying in {delay:.0f}s "
                    f"(state={_state(ec2, instance_id)})"
                )
                time.sleep(delay)
                continue
            raise


def ensure_ssh(ec2, instance_id: str) -> None:
    """Allocate (if none) an Elastic IP, associate it, open SSH on the SG."""
    inst = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    eip = inst.get("PublicIpAddress")
    if eip:
        print(f"already has public IP: {eip}")
        return

    alloc = ec2.allocate_address(Domain="vpc")
    try:
        ec2.associate_address(InstanceId=instance_id, AllocationId=alloc["AllocationId"])
        eip = alloc["PublicIp"]
        print(f"allocated + associated EIP: {eip}")
    except Exception:
        ec2.release_address(AllocationId=alloc["AllocationId"])
        raise

    sg_id = inst["SecurityGroups"][0]["GroupId"]
    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "ssh-zhisa"}],
                }
            ],
        )
        print(f"SSH (22) allowed on {sg_id}")
    except ec2.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "InvalidPermission.Duplicate":
            raise
        print(f"SSH rule already present on {sg_id}")


def wait_running(ec2, instance_id: str, timeout_s: float = 900.0) -> str:
    deadline = time.time() + timeout_s
    printed = set()
    while time.time() < deadline:
        st = _state(ec2, instance_id)
        if st not in printed:
            print(f"state -> {st}")
            printed.add(st)
        if st == "running":
            r = ec2.describe_instances(InstanceIds=[instance_id])
            ip = r["Reservations"][0]["Instances"][0].get("PublicIpAddress")
            if ip:
                return ip
        time.sleep(10)
    print(f"not running within {timeout_s:.0f}s; last state {_state(ec2, instance_id)}")
    sys.exit(3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--backoff", type=float, default=30.0)
    parser.add_argument("--ensure-ssh", action="store_true")
    parser.add_argument("--no-start", action="store_true", help="only report state + IP")
    args = parser.parse_args(argv)

    ec2 = _ec2(args.region, args.profile)

    if args.no_start:
        st = _state(ec2, args.instance_id)
        ip = _public_ip(ec2, args.instance_id)
        print(f"state={st} public_ip={ip}")
        return 0

    start_with_retry(
        ec2, args.instance_id, max_attempts=args.max_attempts, base_backoff=args.backoff
    )
    if args.ensure_ssh:
        ensure_ssh(ec2, args.instance_id)
    ip = wait_running(ec2, args.instance_id, timeout_s=900.0)
    print(f"RUNNING public_ip={ip}")
    print(f"ssh -i C:\\Users\\HP\\.ssh\\zhisa333.pem ubuntu@{ip}")
    return 0


def _public_ip(ec2, instance_id: str) -> str | None:
    r = ec2.describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0].get("PublicIpAddress")


if __name__ == "__main__":
    raise SystemExit(main())