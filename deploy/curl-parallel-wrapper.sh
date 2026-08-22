#!/bin/bash
# curl-parallel: transparent multi-threaded Range downloader wrapper.
#
# Placed at /usr/local/bin/curl (before /usr/bin in PATH). The sglang
# engine's LoRA download path spawns `curl` via subprocess on every
# download, so PATH resolution picks this wrapper up on the NEXT download
# with zero engine restart.
#
# Behavior:
#   - Anything that is not exactly [one https URL] + [-o DEST] passes
#     through to the real curl unchanged (health checks, HEADs, small GETs).
#   - Otherwise: probe Range support (GET 0-0). If 206 + Content-Range and
#     total >= 16MB -> 16-way parallel range download, concatenate, verify
#     exact size, exit 0. Any part failure -> exit 22 (engine marks load
#     failed and retries per its own retry loop).
#   - Server ignores Range (200) or small file -> passthrough single stream.
set -u
REAL=/usr/bin/curl
MIN_PARALLEL_SIZE=16777216
THREADS=16

url=""; dest=""; extra_url=0
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  a="${args[$i]}"
  case "$a" in
    https://*) if [ -n "$url" ]; then extra_url=1; else url="$a"; fi ;;
    -o)  dest="${args[$((i+1))]:-}"; ((i++)) ;;
    --output=*) dest="${a#--output=}" ;;
  esac
done
if [ -z "$url" ] || [ -z "$dest" ] || [ "$extra_url" = "1" ]; then
  exec "$REAL" "$@"
fi

# Probe: range 0-0. max-filesize guards against servers that ignore Range
# (they would stream the entire body as a 200); headers still get captured.
hdr=$("$REAL" -s -r 0-0 --max-filesize 65536 -o /dev/null -D - --max-time 15 "$url" 2>/dev/null || true)
code=$(printf '%s\n' "$hdr" | head -1 | grep -oE '[0-9]{3}' | head -1)
cr=$(printf '%s\n' "$hdr" | tr -d '\r' | grep -i '^content-range:' | tail -1)
if [ "$code" != "206" ] || [ -z "$cr" ]; then
  exec "$REAL" "$@"   # no Range support (or error) -> single stream
fi
total=$(printf '%s' "$cr" | grep -oE '/[0-9]+$' | tr -d '/')
if ! [[ "$total" =~ ^[0-9]+$ ]] || [ "$total" -lt "$MIN_PARALLEL_SIZE" ]; then
  exec "$REAL" "$@"   # small -> not worth it
fi

chunk=$(( (total + THREADS - 1) / THREADS ))
tmpd=$(mktemp -d "${dest}.par.XXXXXX") || exec "$REAL" "$@"
pids=(); used=0
for ((i=0; i<THREADS; i++)); do
  start=$((i*chunk)); [ "$start" -ge "$total" ] && break
  end=$(( start + chunk - 1 )); [ "$end" -ge "$total" ] && end=$((total-1))
  "$REAL" -s -r "${start}-${end}" --retry 5 --retry-delay 2 --retry-all-errors \
    --connect-timeout 10 -o "$(printf '%s/part_%02d' "$tmpd" "$i")" "$url" &
  pids+=($!); used=$((i+1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = "1" ]; then
  rm -rf "$tmpd"
  echo "curl-parallel: a range part failed (url=${url:0:64}...)" >&2
  exit 22
fi
parts=()
for ((i=0; i<used; i++)); do parts+=("$(printf '%s/part_%02d' "$tmpd" "$i")"); done
cat "${parts[@]}" > "$dest" || { rm -rf "$tmpd"; exit 22; }
rm -rf "$tmpd"
got=$(stat -c%s "$dest" 2>/dev/null || echo 0)
if [ "$got" != "$total" ]; then
  echo "curl-parallel: size mismatch got=$got want=$total" >&2
  exit 22
fi
exit 0
