#!/usr/bin/env bash
# launch_8gpu.sh -- spin up 8 per-GPU containers + 8 git worktrees, default 4+4 split.
#
# Usage:
#   bash launcher/launch_8gpu.sh                              # 4 broad_phase + 4 step_1_2
#   bash launcher/launch_8gpu.sh --split 6 2                  # 6 broad_phase + 2 step_1_2
#   bash launcher/launch_8gpu.sh --image genesis:amd-integration --num-gpus 8
#
# After this exits, each container is running detached. The agent (Claude Code / Codex)
# is NOT started -- you start one per worktree manually:
#   for i in 0..7:  cd ~/work/ak-wt/gpu$i  &&  claude code

set -euo pipefail

IMAGE="genesis:amd-integration"
NUM_GPUS=8
N_BROAD=4   # default split
N_STEP=4
AUTOKERNEL_ROOT="${AUTOKERNEL_ROOT:-$HOME/work}"
AK_REPO="${AK_REPO:-$AUTOKERNEL_ROOT/autokernels-genesis}"
GENESIS_REPO="${GENESIS_REPO:-$AUTOKERNEL_ROOT/Genesis}"
QUADRANTS_REPO="${QUADRANTS_REPO:-$AUTOKERNEL_ROOT/quadrants}"
NEWTON_ASSETS="${NEWTON_ASSETS:-$AUTOKERNEL_ROOT/newton-assets}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)         N_BROAD="$2"; N_STEP="$3"; shift 3 ;;
    --image)         IMAGE="$2"; shift 2 ;;
    --num-gpus)      NUM_GPUS="$2"; shift 2 ;;
    --root)    AUTOKERNEL_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if (( N_BROAD + N_STEP != NUM_GPUS )); then
  echo "split must sum to $NUM_GPUS (got $N_BROAD + $N_STEP)" >&2
  exit 1
fi

# Sanity
for d in "$AK_REPO" "$GENESIS_REPO" "$QUADRANTS_REPO" "$NEWTON_ASSETS"; do
  [[ -d "$d" ]] || { echo "missing: $d" >&2; exit 1; }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image $IMAGE not found locally" >&2; exit 1; }

DATE_TAG="$(date +%b%d | tr A-Z a-z)"
mkdir -p "$AUTOKERNEL_ROOT/ak-wt" "$AUTOKERNEL_ROOT/genesis-wt" "$AUTOKERNEL_ROOT/cache"

assign_campaign() {
  local i=$1
  if (( i < N_BROAD )); then echo "func_broad_phase"; else echo "kernel_step_1_2"; fi
}

for ((i=0; i<NUM_GPUS; i++)); do
  CAMPAIGN="$(assign_campaign $i)"
  CONTAINER="ak-gpu${i}"
  AK_WT="$AUTOKERNEL_ROOT/ak-wt/gpu${i}"
  GEN_WT="$AUTOKERNEL_ROOT/genesis-wt/gpu${i}"
  AK_BRANCH="ak/${DATE_TAG}-${CAMPAIGN}-gpu${i}"
  GEN_BRANCH="perf/jlohia/${DATE_TAG}-${CAMPAIGN}-gpu${i}"
  Q_CACHE="$AUTOKERNEL_ROOT/cache/qcache_gpu${i}"
  M_CACHE="$AUTOKERNEL_ROOT/cache/mcache_gpu${i}"

  echo "===== GPU ${i}: ${CAMPAIGN} ====="

  # Worktrees
  if [[ ! -d "$AK_WT" ]]; then
    git -C "$AK_REPO" worktree add -b "$AK_BRANCH" "$AK_WT" 2>/dev/null \
      || git -C "$AK_REPO" worktree add "$AK_WT" "$AK_BRANCH" 2>/dev/null \
      || { echo "  failed to create AK worktree at $AK_WT" >&2; continue; }
    echo "  AK worktree: $AK_WT (branch $AK_BRANCH)"
  else
    echo "  AK worktree exists: $AK_WT"
  fi
  if [[ ! -d "$GEN_WT" ]]; then
    git -C "$GENESIS_REPO" worktree add -b "$GEN_BRANCH" "$GEN_WT" 2>/dev/null \
      || git -C "$GENESIS_REPO" worktree add "$GEN_WT" "$GEN_BRANCH" 2>/dev/null \
      || { echo "  failed to create Genesis worktree at $GEN_WT" >&2; continue; }
    # Genesis expects newton-assets as a sibling/symlink
    ln -sfn "$NEWTON_ASSETS" "$GEN_WT/newton-assets"
    # Copy benchmark_scaling.py if it lives at ~/work/ rather than inside Genesis
    if [[ ! -f "$GEN_WT/benchmark_scaling.py" && -f "$AUTOKERNEL_ROOT/benchmark_scaling.py" ]]; then
      cp "$AUTOKERNEL_ROOT/benchmark_scaling.py" "$GEN_WT/benchmark_scaling.py"
    fi
    echo "  Genesis worktree: $GEN_WT (branch $GEN_BRANCH)"
  else
    echo "  Genesis worktree exists: $GEN_WT"
  fi

  mkdir -p "$Q_CACHE" "$M_CACHE"

  # Container
  if docker inspect -f '{{.State.Running}}' "$CONTAINER" >/dev/null 2>&1; then
    echo "  container $CONTAINER already running"
  else
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -dit \
      --device=/dev/kfd --device=/dev/dri \
      --group-add video --group-add render \
      --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
      --name "$CONTAINER" \
      -e HIP_VISIBLE_DEVICES="$i" \
      -e GS_FAST_MATH=0 \
      -e PYOPENGL_PLATFORM=egl -e EGL_PLATFORM=surfaceless -e PYGLET_HEADLESS=true \
      -e CAMPAIGN="$CAMPAIGN" \
      -e AUTOKERNEL_GPU_ID="$i" \
      -v "$GEN_WT":/src/Genesis \
      -v "$QUADRANTS_REPO":/src/quadrants:ro \
      -v "$NEWTON_ASSETS":/src/newton-assets:ro \
      -v "$AK_WT":/work/autokernels:ro \
      -v "$Q_CACHE":/root/.cache/quadrants \
      -v "$M_CACHE":/root/.cache/mesa_shader_cache \
      -v "$AUTOKERNEL_ROOT/runs":/work/runs \
      "$IMAGE" sleep infinity >/dev/null
    docker exec "$CONTAINER" bash -lc \
      'ln -sfn /src/newton-assets /src/Genesis/newton-assets || true' >/dev/null
    echo "  container $CONTAINER started, pinned to GPU $i"
  fi
done

echo
echo "Done. To start agents (one per worktree):"
for ((i=0; i<NUM_GPUS; i++)); do
  CAMPAIGN="$(assign_campaign $i)"
  AK_WT="$AUTOKERNEL_ROOT/ak-wt/gpu${i}"
  echo "  GPU ${i} (${CAMPAIGN}): cd $AK_WT && \\"
  echo "    AUTOKERNEL_GPU_ID=$i AUTOKERNEL_CONTAINER=ak-gpu${i} \\"
  echo "    GENESIS_SRC=$AUTOKERNEL_ROOT/genesis-wt/gpu${i} CAMPAIGN=$CAMPAIGN \\"
  echo "    claude code   # or: codex"
done
