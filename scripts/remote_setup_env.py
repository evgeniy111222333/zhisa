import subprocess

KEY = r'C:\Users\HP\.ssh\zhisa333.pem'
IP = '16.171.168.229'
USER = 'ubuntu'

def ssh_run(command):
    cmd = ['ssh', '-i', KEY, '-o', 'StrictHostKeyChecking=no', f'{USER}@{IP}', command]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr

if __name__ == '__main__':
    script = """
    echo "=== INSTALLED PYTHON PACKAGES & BINARIES ==="
    ls /usr/bin/python* || true
    which pip || which pip3 || true
    sudo apt update -y
    sudo apt install -y python3-pip python3-venv python3-dev build-essential rsync tmux htop
    """
    code, out, err = ssh_run(script)
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
