#!/usr/bin/env bash
# docker_run.sh -- single-container template (e.g. for prepare.py or smoke testing GPU 0).
#
# Usage: GPU_ID=0 CAMPAIGN=func_broad_phase bash launcher/docker_run.sh
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
IMAGE="${IMAGE:-genesis:amd-integration}"
AUTOKERNEL_ROOT="${AUTOKERNEL_ROOT:-$HOME/work}"
NAME="${CONTAINER_NAME:-ak-gpu${GPU_ID}}"
CAMPAIGN="${CAMPAIGN:-func_broad_phase}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -dit \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
  --name "$NAME" \
  -e HIP_VISIBLE_DEVICES="$GPU_ID" \
  -e GS_FAST_MATH=0 \
  -e PYOPENGL_PLATFORM=egl -e EGL_PLATFORM=surfaceless -e PYGLET_HEADLESS=true \
  -e CAMPAIGN="$CAMPAIGN" \
  -e AUTOKERNEL_GPU_ID="$GPU_ID" \
  -v "$AUTOKERNEL_ROOT/Genesis":/src/Genesis \
  -v "$AUTOKERNEL_ROOT/quadrants":/src/quadrants:ro \
  -v "$AUTOKERNEL_ROOT/newton-assets":/src/newton-assets:ro \
  -v "$AUTOKERNEL_ROOT/runs":/work/runs \
  "$IMAGE" sleep infinity

docker exec "$NAME" bash -lc \
  'ln -sfn /src/newton-assets /src/Genesis/newton-assets || true'

echo "container $NAME up, pinned to GPU $GPU_ID, campaign=$CAMPAIGN"
echo "to bench: AUTOKERNEL_GPU_ID=$GPU_ID AUTOKERNEL_CONTAINER=$NAME \\"
echo "          GENESIS_SRC=$AUTOKERNEL_ROOT/Genesis \\"
echo "          uv run bench.py --campaign $CAMPAIGN"
