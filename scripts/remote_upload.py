import subprocess
import os

KEY = r'C:\Users\HP\.ssh\zhisa333.pem'
IP = '16.171.168.229'
USER = 'ubuntu'
LOCAL_TAR = r'd:\zhisa\upload_s2b_package.tar.gz'
REMOTE_TAR = '/opt/dlami/nvme/upload_s2b_package.tar.gz'

def upload():
    print(f"Uploading {LOCAL_TAR} ({os.path.getsize(LOCAL_TAR)/(1024*1024):.2f} MB) to {USER}@{IP}:{REMOTE_TAR}...")
    cmd = [
        'scp',
        '-i', KEY,
        '-o', 'StrictHostKeyChecking=no',
        LOCAL_TAR,
        f'{USER}@{IP}:{REMOTE_TAR}'
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("SCP failed:", res.stderr)
        return False
    print("SCP finished successfully!")
    return True

if __name__ == '__main__':
    upload()
