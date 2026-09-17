#!/bin/bash
# ===========================================================================
# v4 镜像重建（**本地构建，不 push registry**）+ 逐项自检
#
#   v4 = v3 基础镜像 + 目标 commit 的 sglang 代码（叠加，不重编 mooncake/sgl-kernel）
#   跑完得到本地 tag：mindverse/sglang:<tag>，分发用 docker save/load（README §7）
#
# 用法：
#   bash rebuild-image.sh                                  # 用当前 HEAD
#   bash rebuild-image.sh --ref origin/b300-glm52
#   bash rebuild-image.sh --ref <sha> --tag my-sglang:v4
#   bash rebuild-image.sh --with-p2p-ar                    # 额外编译 #24 的 P2P AR 扩展
#
# 自检（任一不过即失败退出）：
#   1. 构建期：import 路径指向源码树、compileall 通过、v3 的构建产物仍在、mooncake engine.so md5 未变
#   2. 构建后：镜像里 python/ 全量文件的 sha256 == 该 commit 的树（证明 COPY 完整、无残留旧文件）
#   3. 构建后：探针逐项与基础镜像对比（.so 数量、/opt/p2p_ar、jit .cuh 数量不减少）
# ===========================================================================
set -euo pipefail

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(git -C "$SELF_DIR" rev-parse --show-toplevel)
BASE_IMAGE="b200routeraca.azurecr.io/mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v3"
REF="HEAD"
TAG=""
WITH_P2P_AR=0
CTX=""

while [ $# -gt 0 ]; do
    case "$1" in
        --ref)        REF="$2"; shift 2 ;;
        --base)       BASE_IMAGE="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --with-p2p-ar) WITH_P2P_AR=1; shift ;;
        --context)    CTX="$2"; shift 2 ;;
        -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SHA=$(git -C "$REPO_ROOT" rev-parse --verify "$REF^{commit}")
SHORT=$(git -C "$REPO_ROOT" rev-parse --short=8 "$SHA")
[ -n "$TAG" ] || TAG="mindverse/sglang:v0.5.15.post1-cuda13-b200-glm53-final-v4-$SHORT"
[ -n "$CTX" ] || CTX="${TMPDIR:-/tmp}/sglang-rebuild-$SHORT"

echo "== v4 镜像重建 =="
echo "   仓库     : $REPO_ROOT"
echo "   目标 ref : $REF -> $SHA"
echo "   基础镜像 : $BASE_IMAGE"
echo "   本地 tag : $TAG      (不会 push)"
echo "   构建目录 : $CTX"
echo

# ---- 0. 前置检查 ----------------------------------------------------------
command -v docker >/dev/null || { echo "!! 需要 docker" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "!! docker daemon 不可用" >&2; exit 1; }
if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    echo "!! 警告：工作区不干净；本脚本只用 git archive 出的 $SHA 内容，未提交改动不会进镜像"
fi
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
    echo "== 拉取基础镜像（需要 registry 权限）=="; docker pull "$BASE_IMAGE"; }

# ---- 1. 构建上下文 = git archive 该 commit --------------------------------
rm -rf "$CTX"; mkdir -p "$CTX/sglang-src"
git -C "$REPO_ROOT" archive --format=tar "$SHA" | tar -x -C "$CTX/sglang-src"
cp "$SELF_DIR/Dockerfile.rebuild" "$CTX/"
printf 'sglang-src/.git\nsglang-src/**/__pycache__\nsglang-src/**/*.pyc\n' > "$CTX/.dockerignore"

# 二进制不该出现在 git 树里（出现说明 ref 不对，或有人把产物提交了）
if find "$CTX/sglang-src" -type f \( -name '*.so' -o -name '*.a' \) | head -1 | grep -q .; then
    echo "!! 构建上下文里出现 *.so/*.a —— 拒绝继续（会遮掉镜像里的编译产物）" >&2
    exit 1
fi

# ---- 2. 镜内探针（基础镜像 vs 新镜像用同一份） -----------------------------
cat > "$CTX/probe.sh" <<'PROBE'
#!/bin/sh
# 镜内自检探针：宿主机与镜像里跑同一份，输出 KEY=VALUE 供逐项对比
ROOT=/sgl-workspace/sglang
echo "sglang_origin=$(python3 -c 'import importlib.util as u; print(u.find_spec("sglang").origin)')"
echo "so_count=$(find $ROOT/python -name '*.so' | wc -l)"
echo "has_version_py=$([ -f $ROOT/python/sglang/_version.py ] && echo yes || echo no)"
echo "has_egg_info=$([ -f $ROOT/python/sglang.egg-info/PKG-INFO ] && echo yes || echo no)"
echo "jit_cuh_count=$(find $ROOT/python/sglang/jit_kernel -name '*.cuh' 2>/dev/null | wc -l)"
echo "mooncake_md5=$(md5sum /usr/local/lib/python3.12/dist-packages/mooncake/engine.so 2>/dev/null | cut -d' ' -f1)"
echo "p2p_ar_so=$(ls /opt/p2p_ar/*.so 2>/dev/null | wc -l)"
PROBE
chmod +x "$CTX/probe.sh"

echo "== 2. 记录基础镜像基线 =="
docker run --rm -v "$CTX/probe.sh:/probe.sh:ro" --entrypoint /probe.sh "$BASE_IMAGE" \
    | tee "$CTX/probe-base.txt"
echo

# ---- 3. 构建 --------------------------------------------------------------
echo "== 3. docker build（不 push）=="
docker build -f "$CTX/Dockerfile.rebuild" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "SGLANG_REF=$SHA" \
    --build-arg "WITH_P2P_AR=$WITH_P2P_AR" \
    -t "$TAG" "$CTX"
echo

# ---- 4. 自检：代码清单一比一 / 探针逐项对比 -------------------------------
echo "== 4. 自检 =="
echo "   (a) 镜像里 python/ 的 sha256 清单 vs commit $SHORT 的树"
# 注意 -print0 / xargs -0：树里有含空格的路径（如 ".../block_shape=[128, 128].json"）
docker run --rm --entrypoint sh "$TAG" -c \
    'cd /sgl-workspace/sglang && find python -type f ! -path "*__pycache__*" ! -name "*.pyc" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum' \
    > "$CTX/image.txt"
(cd "$CTX/sglang-src" && find python -type f ! -path "*__pycache__*" ! -name "*.pyc" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) \
    > "$CTX/src.txt"

python3 - "$CTX/src.txt" "$CTX/image.txt" <<'PY'
import sys
from pathlib import Path
def load(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            d[parts[1].strip()] = parts[0]
    return d
src, img = load(sys.argv[1]), load(sys.argv[2])
missing = [k for k in src if k not in img]
differ = [k for k in src if k in img and src[k] != img[k]]
extra = [k for k in img if k not in src]
print(f"   源文件 {len(src)} / 镜像里 {len(img)}")
print(f"   缺失 {len(missing)} · 不一致 {len(differ)} · 镜像多出 {len(extra)}")
for label, items in (("缺失", missing), ("不一致", differ)):
    for k in items[:10]:
        print(f"     !! {label}: {k}")
if missing or differ:
    print("    !! 代码树与目标 commit 不一致 —— 镜像不可用")
    sys.exit(1)
print("    ✓ 代码与目标 commit 逐字节一致（镜像多出的文件 = v3 构建产物，符合预期）")
PY

echo "   (b) 探针 vs 基础镜像"
docker run --rm -v "$CTX/probe.sh:/probe.sh:ro" --entrypoint /probe.sh "$TAG" | tee "$CTX/probe-new.txt"
python3 - "$CTX/probe-base.txt" "$CTX/probe-new.txt" "$WITH_P2P_AR" <<'PY'
import sys
from pathlib import Path
def load(p):
    return dict(l.split("=", 1) for l in Path(p).read_text().splitlines() if "=" in l)
base, new, want_p2p = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3] == "1"
fails = []
if "/sgl-workspace/sglang/python/sglang/" not in new.get("sglang_origin", ""):
    fails.append(f"import 路径不对: {new.get('sglang_origin')}")
if new.get("so_count") != base.get("so_count"):
    fails.append(f".so 数量变了: base={base.get('so_count')} new={new.get('so_count')}（可能遮掉了编译产物）")
for k in ("has_version_py", "has_egg_info"):
    if new.get(k) != "yes":
        fails.append(f"{k}={new.get(k)}（v3 构建产物丢失）")
if int(new.get("jit_cuh_count", 0)) < int(base.get("jit_cuh_count", 0)):
    fails.append(f"jit .cuh 变少: {base.get('jit_cuh_count')} -> {new.get('jit_cuh_count')}")
if new.get("mooncake_md5") != base.get("mooncake_md5"):
    fails.append(f"mooncake engine.so md5 变了: {base.get('mooncake_md5')} -> {new.get('mooncake_md5')}")
if want_p2p:
    if new.get("p2p_ar_so", "0") == "0":
        fails.append("--with-p2p-ar 指定了但 /opt/p2p_ar 里没有 .so")
elif new.get("p2p_ar_so", "0") != "0":
    fails.append("未指定 --with-p2p-ar 却出现了 p2p_ar .so")
if fails:
    print("    !! 探针不通过：")
    for f in fails:
        print("       -", f)
    sys.exit(1)
print("    ✓ 探针全部通过（构建产物保留、编译产物未被遮、mooncake 未变）")
if want_p2p:
    print("    ℹ️  已编译 p2p_ar：启用需要 SGLANG_P2P_AR=1 + SGLANG_P2P_AR_LIB=/opt/p2p_ar/libp2p_ar.so（该功能默认关，未在本机验证）")
PY

echo
echo "== 完成 =="
docker images --format '  {{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}' "$TAG"
cat <<EOF

下一步（分发到机器，**不要 push**）：
  # 1) 本机导出（约几分钟）
  docker save "$TAG" | zstd -T0 -3 > /tmp/$(echo "$TAG" | tr '/:' '__').tar.zst
  # 2) 传到两台机器并导入（decode / prefill 都要）
  scp /tmp/$(echo "$TAG" | tr '/:' '__').tar.zst ubuntu@15.164.0.39:/tmp/
  ssh ubuntu@15.164.0.39 'zstd -dc /tmp/*.tar.zst | docker load'
  scp /tmp/$(echo "$TAG" | tr '/:' '__').tar.zst ubuntu@3.37.20.114:/tmp/
  ssh ubuntu@3.37.20.114 'zstd -dc /tmp/*.tar.zst | docker load'
  # 3) 切换（两端一起，PD 铁律）：start_mol_{prefill,decode}.sh 里改 LMS_ENGINE_IMAGE
  #    LMS_ENGINE_IMAGE=$TAG   # 或写进 /etc/lora2endpoint.env 供脚本读取
  #    bash /root/mol-stack.sh restart
  # 4) 回滚：把 LMS_ENGINE_IMAGE 换回 v3 tag（v3 镜像仍在机器上）+ 配对重启
EOF
