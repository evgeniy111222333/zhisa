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
    set -e
    echo "=== INSTALLING PYTHON 3.12 & DEPENDENCIES ==="
    sudo add-apt-repository -y ppa:deadsnakes/ppa || true
    sudo apt update -y
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
    
    echo "=== RECREATING VENV WITH PYTHON 3.12 ==="
    rm -rf /opt/dlami/nvme/venv
    python3.12 -m venv /opt/dlami/nvme/venv
    
    source /opt/dlami/nvme/venv/bin/activate
    python -V
    pip install --upgrade pip setuptools wheel
    
    echo "=== INSTALLING PYTORCH WITH CUDA 12.4+ ==="
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    python -c "
import torch
print('>>> SUCCESS! PyTorch version:', torch.__version__)
print('>>> CUDA available:', torch.cuda.is_available())
print('>>> GPU Device:', torch.cuda.get_device_name(0))
"
    
    echo "=== INSTALLING PROJECT DEPENDENCIES ==="
    pip install numpy pandas scipy matplotlib pyyaml tqdm pyarrow gymnasium optuna ccxt websockets pytest
    """
    code, out, err = ssh_run(script)
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
