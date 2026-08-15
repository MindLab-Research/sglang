#!/bin/bash
# 修改 1P1D start_pd.sh 部署融合 L2 模型（去 LoRA 虚拟专家）
# 用法: bash prep_fused_deploy.sh <node_port>   (1021=prefill, 1022=decode)
PORT=$1
if [ -z "$PORT" ]; then echo "usage: $0 <1021|1022>"; exit 1; fi

ssh -p $PORT -o ConnectTimeout=15 root@8.213.215.2 bash -s << 'EOF'
set -e
cd /root
cp start_pd.sh start_pd.sh.bak_mol  # 备份 MoL 版

# 1. MODEL_PATH → 融合模型
sed -i 's|MODEL_PATH="/root/glm52_local/base"|MODEL_PATH="/root/glm52_local/base_l2_merged"|' start_pd.sh

# 2. 去掉 LoRA 虚拟专家参数（prefill + decode 两个函数）
sed -i '/--enable-lora/d' start_pd.sh
sed -i '/--lora-paths L0/d' start_pd.sh
sed -i '/--max-lora-rank 16/d' start_pd.sh
sed -i '/--max-loaded-loras 4/d' start_pd.sh
sed -i '/--max-loras-per-batch 4/d' start_pd.sh
sed -i '/--lora-use-virtual-experts/d' start_pd.sh
sed -i '/--max-lora-chunk-size 128/d' start_pd.sh

echo "=== 修改后 model-path / lora / eagle 配置 ==="
grep -E 'MODEL_PATH|model-path|enable-lora|lora-paths|speculative|draft-model-path' start_pd.sh | head -10
echo "=== 确认无 lora 残留 ==="
grep -c 'enable-lora\|lora-paths\|virtual-experts' start_pd.sh || echo "0 (clean)"
EOF