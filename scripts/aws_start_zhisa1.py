"""Auto-start zhisa1 (i-016771ecc77ee6bf6) — retry loop for capacity backoff.

Runs on a schedule; if the instance is already running it exits quietly, so the
task is harmless when it eventually succeeds. Logs to the temp dir.
"""
import time
import boto3

IID = "i-016771ecc77ee6bf6"
REGION = "eu-north-1"
LOG = r"C:\Users\HP\AppData\Local\Temp\opencode\aws_start.log"
ATTEMPTS = 8
DELAY = 30

def main() -> int:
    ec2 = boto3.client("ec2", region_name=REGION)
    with open(LOG, "a", encoding="utf-8") as f:
        def log(m):
            print(m, flush=True)
            f.write(f"{time.strftime('%H:%M:%S')} {m}\n")

        state = ec2.describe_instances(InstanceIds=[IID])["Reservations"][0]["Instances"][0]["State"]["Name"]
        if state in ("running", "pending"):
            log(f"already {state} - nothing to do")
            return 0
        for attempt in range(1, ATTEMPTS + 1):
            try:
                r = ec2.start_instances(InstanceIds=[IID])
                log(f"attempt {attempt} OK -> {r['StartingInstances'][0]['CurrentState']['Name']}")
                return 0
            except Exception as e:
                code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
                log(f"attempt {attempt} {code}")
                if attempt < ATTEMPTS:
                    time.sleep(DELAY)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())