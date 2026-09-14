#!/bin/bash
# patch_mooncake_hip_transport.sh
# 禁用 mooncake HIP IPC transport, 强制使用 RDMA transport
#
# 根因: mooncake 编译时 -DUSE_HIP=ON -DENABLE_MULTI_PROTOCOL=ON
#   注册了 HIP IPC transport (priority=4) 和 RDMA transport (priority=2)
#   HIP priority > RDMA, 跨节点传输错误选择 HIP IPC (只支持同节点)
#   导致 hipIpcOpenMemHandle failed → segfault
#
# Patch:
#   1. transfer_engine_impl.cpp: #ifdef USE_HIP → #ifdef USE_HIP_DISABLED (跳过 HIP transport 注册)
#   2. multi_transport.cpp: return 4 → return 0 (HIP priority 降为 0, RDMA 优先)
#
# 用法: 在容器内执行
#   bash patch_mooncake_hip_transport.sh          # apply
#   bash patch_mooncake_hip_transport.sh --check  # dry-run
#   bash patch_mooncake_hip_transport.sh --revert  # revert

set -ex

MOONCAKE_SRC="/sgl-workspace/Mooncake/mooncake-transfer-engine/src"
IMPL_FILE="${MOONCAKE_SRC}/transfer_engine_impl.cpp"
TRANSPORT_FILE="${MOONCAKE_SRC}/multi_transport.cpp"

MODE="${1:-apply}"

# 检查文件存在
if [ ! -f "$IMPL_FILE" ]; then
    echo "ERROR: $IMPL_FILE not found"
    exit 1
fi
if [ ! -f "$TRANSPORT_FILE" ]; then
    echo "ERROR: $TRANSPORT_FILE not found"
    exit 1
fi

# 备份
if [ "$MODE" = "apply" ] && [ ! -f "${IMPL_FILE}.orig" ]; then
    cp "$IMPL_FILE" "${IMPL_FILE}.orig"
    cp "$TRANSPORT_FILE" "${TRANSPORT_FILE}.orig"
    echo "Backed up to .orig"
fi

case "$MODE" in
    apply)
        # Patch 1: transfer_engine_impl.cpp — 禁用 HIP transport 注册
        # 搜索 #ifdef USE_HIP 在 transport 注册上下文中, 替换为 USE_HIP_DISABLED
        if grep -q "#ifdef USE_HIP\b" "$IMPL_FILE"; then
            echo "Patching transfer_engine_impl.cpp: #ifdef USE_HIP → #ifdef USE_HIP_DISABLED"
            sed -i 's/#ifdef USE_HIP\b/#ifdef USE_HIP_DISABLED/g' "$IMPL_FILE"
            echo "  Patched $(grep -c 'USE_HIP_DISABLED' "$IMPL_FILE") occurrences"
        else
            echo "transfer_engine_impl.cpp: already patched or no match"
        fi

        # Patch 2: multi_transport.cpp — HIP priority 4 → 0
        # 搜索 HIP transport 的 return 4 (priority), 替换为 return 0
        if grep -q "return 4" "$TRANSPORT_FILE"; then
            echo "Patching multi_transport.cpp: HIP transport priority 4 → 0"
            # 只替换 HIP transport 相关的 return 4
            # 上下文: HipTransport::priority() { return 4; }
            sed -i '/HipTransport\|hip_transport\|USE_HIP/{s/return 4;/return 0;/g}' "$TRANSPORT_FILE"
            echo "  Patched"
        else
            echo "multi_transport.cpp: already patched or no match"
        fi

        echo ""
        echo "=== Patch applied ==="
        echo "Verify:"
        grep -n "USE_HIP_DISABLED" "$IMPL_FILE" | head -5
        grep -n "return 0" "$TRANSPORT_FILE" | grep -i "hip\|priority" | head -5
        ;;

    --check)
        echo "=== Dry run ==="
        echo "transfer_engine_impl.cpp:"
        grep -n "#ifdef USE_HIP\b" "$IMPL_FILE" | head -5
        echo "multi_transport.cpp:"
        grep -n "return 4" "$TRANSPORT_FILE" | head -5
        ;;

    --revert)
        if [ -f "${IMPL_FILE}.orig" ]; then
            cp "${IMPL_FILE}.orig" "$IMPL_FILE"
            cp "${TRANSPORT_FILE}.orig" "$TRANSPORT_FILE"
            echo "Reverted from .orig"
        else
            echo "No .orig backup found"
            exit 1
        fi
        ;;

    *)
        echo "Usage: $0 [apply|--check|--revert]"
        exit 1
        ;;
esac
