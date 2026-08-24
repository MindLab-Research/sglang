#!/bin/bash
# patch_mooncake_dmabuf.sh
# 启用 GPUDirect RDMA (dmabuf) for mooncake on AMD MI300X
#
# 3 处修改:
#   1. transfer_engine_impl.cpp: #ifdef USE_HIP → #ifdef USE_HIP_DISABLED (禁用 HIP IPC transport, 修复 segfault)
#   2. multi_transport.cpp: return 4 → return 0 (HIP priority 降为 0, 强制用 RDMA)
#   3. rdma_transport/CMakeLists.txt: 添加 target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
#
# 前提:
#   - v0.5.16 镜像中 mooncake 源码在 /sgl-workspace/Mooncake/
#   - CMakeCache 已有 USE_HIP_DMABUF:BOOL=ON (但 rdma_transport target 没有传递)
#   - 内核支持 CONFIG_PCI_P2PDMA=y + CONFIG_DMABUF_MOVE_NOTIFY=y
#
# 用法: 在容器内执行
#   bash patch_mooncake_dmabuf.sh          # apply + 编译
#   bash patch_mooncake_dmabuf.sh --check  # 只检查
#   bash patch_mooncake_dmabuf.sh --revert # 回退

set -ex

MOONCAKE_SRC="/sgl-workspace/Mooncake/mooncake-transfer-engine/src"
BUILD_DIR="/sgl-workspace/Mooncake/build"
MODE="${1:-apply}"

# 检查文件存在
for f in \
    "$MOONCAKE_SRC/transfer_engine_impl.cpp" \
    "$MOONCAKE_SRC/multi_transport.cpp" \
    "$MOONCAKE_SRC/transport/rdma_transport/CMakeLists.txt"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

case "$MODE" in
    --check)
        echo "=== Patch 1: transfer_engine_impl.cpp ==="
        grep -n "USE_HIP_DISABLED" "$MOONCAKE_SRC/transfer_engine_impl.cpp" | head -3
        echo "=== Patch 2: multi_transport.cpp ==="
        grep -n "return 0" "$MOONCAKE_SRC/multi_transport.cpp" | grep "p ==" | head -3
        echo "=== Patch 3: rdma_transport CMakeLists ==="
        grep -n "USE_HIP_DMABUF" "$MOONCAKE_SRC/transport/rdma_transport/CMakeLists.txt" | head -3
        echo "=== CMakeCache ==="
        grep USE_HIP_DMABUF "$BUILD_DIR/CMakeCache.txt" 2>/dev/null
        ;;

    --revert)
        for f in transfer_engine_impl.cpp multi_transport.cpp; do
            if [ -f "$MOONCAKE_SRC/$f.orig" ]; then
                cp "$MOONCAKE_SRC/$f.orig" "$MOONCAKE_SRC/$f"
                echo "Reverted $f"
            fi
        done
        # CMakeLists 需要手动删除最后一行
        sed -i '/target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)/d' \
            "$MOONCAKE_SRC/transport/rdma_transport/CMakeLists.txt"
        echo "Reverted CMakeLists.txt"
        ;;

    apply)
        # 备份
        cp "$MOONCAKE_SRC/transfer_engine_impl.cpp" "$MOONCAKE_SRC/transfer_engine_impl.cpp.orig" 2>/dev/null || true
        cp "$MOONCAKE_SRC/multi_transport.cpp" "$MOONCAKE_SRC/multi_transport.cpp.orig" 2>/dev/null || true

        # Patch 1: 禁用 HIP transport 注册
        sed -i 's/#ifdef USE_HIP\b/#ifdef USE_HIP_DISABLED/g' "$MOONCAKE_SRC/transfer_engine_impl.cpp"
        echo "Patch 1: transfer_engine_impl.cpp (#ifdef USE_HIP → USE_HIP_DISABLED)"
        grep -c "USE_HIP_DISABLED" "$MOONCAKE_SRC/transfer_engine_impl.cpp"

        # Patch 2: HIP priority 4 → 0
        sed -i 's/return 4;/return 0;/g' "$MOONCAKE_SRC/multi_transport.cpp"
        echo "Patch 2: multi_transport.cpp (return 4 → return 0)"

        # Patch 3: USE_HIP_DMABUF 加到 rdma_transport target
        if ! grep -q "target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)" \
            "$MOONCAKE_SRC/transport/rdma_transport/CMakeLists.txt"; then
            echo 'target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)' >> \
                "$MOONCAKE_SRC/transport/rdma_transport/CMakeLists.txt"
            echo "Patch 3: rdma_transport/CMakeLists.txt (added USE_HIP_DMABUF)"
        else
            echo "Patch 3: already applied"
        fi

        # 重新 cmake
        cd "$BUILD_DIR"
        cmake .. 2>&1 | tail -3

        # 删除 .o 强制重新编译
        rm -f mooncake-transfer-engine/src/CMakeFiles/transfer_engine.dir/transfer_engine_impl.cpp.o
        rm -f mooncake-transfer-engine/src/CMakeFiles/transfer_engine.dir/multi_transport.cpp.o
        rm -f mooncake-transfer-engine/src/transport/rdma_transport/CMakeFiles/rdma_transport.dir/rdma_context.cpp.o
        rm -f mooncake-integration/engine.cpython-310-x86_64-linux-gnu.so

        # 编译
        make engine -j$(nproc) 2>&1 | tail -10
        echo "BUILD_EXIT=$?"

        # 验证
        echo "=== md5 ==="
        md5sum mooncake-integration/engine.cpython-310-x86_64-linux-gnu.so
        echo "=== dmabuf strings ==="
        strings mooncake-integration/engine.cpython-310-x86_64-linux-gnu.so | grep -iE "hsa_amd_portable|ibv_reg_dmabuf|isKernelDmabuf" | head -5
        echo "=== done ==="
        ;;

    *)
        echo "Usage: $0 [apply|--check|--revert]"
        exit 1
        ;;
esac
