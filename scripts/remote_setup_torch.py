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
    # Give user write access to /opt/dlami/nvme
    sudo chown -R ubuntu:ubuntu /opt/dlami/nvme
    
    mkdir -p /opt/dlami/nvme/zhisa /opt/dlami/nvme/venv
    
    # Try creating venv with python3.12 or python3.11 if present, else python3
    if command -v python3.12 &> /dev/null; then
        python3.12 -m venv /opt/dlami/nvme/venv
    elif command -v python3.11 &> /dev/null; then
        python3.11 -m venv /opt/dlami/nvme/venv
    else
        python3 -m venv /opt/dlami/nvme/venv
    fi
    
    source /opt/dlami/nvme/venv/bin/activate
    python -V
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || pip install torch torchvision
    
    # Verify PyTorch & CUDA
    python -c "import torch; print('PyTorch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    
    # Install core libraries
    pip install numpy pandas scipy matplotlib pyyaml tqdm pyarrow gymnasium optuna ccxt websockets pytest
    """
    code, out, err = ssh_run(script)
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
