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
    echo "=== INSTALLING UV FOR ULTRAFAST CPYTHON 3.12 & PYTORCH ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.local/bin/env || export PATH="$HOME/.local/bin:$PATH"
    
    echo "=== CREATING PYTHON 3.12 VENV VIA UV ==="
    rm -rf /opt/dlami/nvme/venv
    uv venv /opt/dlami/nvme/venv --python 3.12
    
    source /opt/dlami/nvme/venv/bin/activate
    python -V
    
    echo "=== INSTALLING PYTORCH WITH CUDA ==="
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    python -c "
import torch
print('>>> SUCCESS! PyTorch version:', torch.__version__)
print('>>> CUDA available:', torch.cuda.is_available())
print('>>> GPU Device:', torch.cuda.get_device_name(0))
"
    
    echo "=== INSTALLING ZHISA DEPENDENCIES ==="
    uv pip install numpy pandas scipy matplotlib pyyaml tqdm pyarrow gymnasium optuna ccxt websockets pytest
    """
    code, out, err = ssh_run(script)
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
