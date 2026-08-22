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
    mkdir -p /opt/dlami/nvme/zhisa
    cd /opt/dlami/nvme/zhisa
    
    echo "=== EXTRACTING PACKAGE ==="
    tar -xzf /opt/dlami/nvme/upload_s2b_package.tar.gz
    
    export PATH="/home/ubuntu/.local/bin:/opt/dlami/nvme/venv/bin:$PATH"
    
    echo "=== INSTALLING ZHISA IN VENV VIA UV ==="
    uv pip install -e /opt/dlami/nvme/zhisa --python /opt/dlami/nvme/venv/bin/python
    
    echo "=== VERIFYING IMPORTS & HARDWARE ==="
    /opt/dlami/nvme/venv/bin/python -c "
import torch
import zhisa
from zhisa.models.policy import PolicyNetwork
from zhisa.data.preparation import load_prepared_split
print('>>> PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))
print('>>> ZHISA package loaded successfully!')
"
    
    mkdir -p artifacts/s2b
    
    echo "=== STARTING S2b TRAINING IN TMUX SESSION ==="
    tmux kill-session -t s2b_training 2>/dev/null || true
    
    # Launch inside tmux session with logging
    tmux new-session -d -s s2b_training "bash -c '
        cd /opt/dlami/nvme/zhisa
        export PATH=\"/home/ubuntu/.local/bin:/opt/dlami/nvme/venv/bin:\$PATH\"
        export ZHISA_FAST_RENDER=1
        export PYTHONUNBUFFERED=1
        /opt/dlami/nvme/venv/bin/python -m zhisa.scripts.train_s2b \
            --config configs/s2b_v2_15m_12markets.yaml \
            --s2-checkpoint artifacts/s2_mtf_champions/s2_mtf_lrfix_best_guarded_20260625.pt \
            --prepared-root data/prepared/s1_15m_12m_v2 \
            --checkpoint artifacts/s2b/s2b_mtf_champion_v2.pt \
            --fast-render \
            --device cuda 2>&1 | tee s2b_train.log
    '"
    
    echo "Tmux session 's2b_training' started!"
    sleep 2
    tmux list-sessions
    """
    
    code, out, err = ssh_run(script)
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
