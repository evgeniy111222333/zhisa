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
    echo "=== PYTHON VERSIONS ==="
    which python3
    python3 --version
    
    echo "=== CONDA / VENVS ==="
    if command -v conda &> /dev/null; then
        conda info --envs
    else
        echo "No conda found in default PATH"
    fi
    
    find / -maxdepth 3 -name "activate" 2>/dev/null || true
    
    echo "=== PYTORCH CHECK ==="
    python3 -c "
try:
    import torch
    print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print('Torch import error:', e)
"
    
    echo "=== DISK MOUNTS ==="
    df -h
    ls -la /opt/dlami/nvme || true
    """
    
    code, out, err = ssh_run(script)
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
