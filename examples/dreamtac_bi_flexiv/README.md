# Dream-Tac on BiFlexiv Rizon4 RT

This is the robot-side half of the Dream-Tac deployment. The robot computer
does not load OpenPI or Dream-Tac model weights: LeRobot-Xense owns the arms,
grippers and seven cameras, while `xense-client` talks to a Dream-Tac WebSocket
server on a separate GPU computer.

## Data path

```text
LeRobot-Xense recipe -> BiFlexivRizon4RTConfig
  20D state + head/wrists + four tactile frames
        |
        v
DreamTacBiFlexivEnvironment
  direct 224x224 resize + consecutive-frame tactile gate
        |
        v
DreamTacRemotePolicy -> Dream-Tac WebSocket server
        |
        v
(30, 20) absolute action chunk -> robot.send_action()
```

The state/action order is:

```text
[left_tcp(0:9), right_tcp(9:18), left_gripper(18), right_gripper(19)]
```

Dream-Tac requires exactly seven views: `head`, both wrists and two tactile
cameras per gripper. `images_raw` is never sent to the inference server; when
`--args.include-raw-images` is enabled it stays in the local observation.
The three RGB views are resized to `224x224` on the robot computer. The four
raw `400x700` tactile views are sent without resizing because the inference
server must merge each sensor pair before applying the training-time resize
and padding.

## Robot recipe

`bi_mount_type` and the old station table no longer exist in LeRobot-Xense.
Pass `--args.robot-recipe` instead. A recipe contains the arm serial numbers,
start/home poses, head camera and typed gripper block.

Prefer an explicit path to the current recipe in the LeRobot-Xense checkout:

```bash
ls /path/to/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/
```

Choose the file that matches both the physical bench and gripper family, for
example `forward-05-taccap.yaml` or `forward-05-xgripper.yaml`. Before running,
verify the arm serial numbers and ensure its gripper block contains:

```yaml
robot:
  gripper:
    auto_discover_cameras: true
    enable_tactile: true
```

Dream-Tac pins `use_force: false`: its action vector has no wrench dimensions.
A recipe with only one gripper, no head camera, or neither camera discovery nor
seven explicitly pinned views is rejected before the policy client or hardware
connection starts.

## Robot-computer installation

Create/install the LeRobot-Xense environment using that repository's installer.
For a Flexiv bench with TacCap grippers:

```bash
cd /path/to/lerobot-xense
git submodule update --init --recursive
bash setup_env.sh --mamba lerobot-xense       # one-time environment creation
mamba activate lerobot-xense
bash setup_env.sh --install --flexiv --taccap
```

For serial/XGripper hardware, `--install --flexiv` installs the Flexiv and Xense
stack. Then install only the lightweight OpenPI robot client and CLI helpers:

```bash
python -m pip install -e /path/to/xense-openpi/packages/xense-client
python -m pip install "tyro>=0.9.5" dm-env pyyaml
```

Confirm that the editable package is the intended LeRobot-Xense checkout:

```bash
python - <<'PY'
import lerobot
import xense_client
import flexiv_rt

print("lerobot:", lerobot.__file__)
print("xense_client:", xense_client.__file__)
print("flexiv_rt:", flexiv_rt.__file__)
PY
```

## Start the inference server

On the GPU computer, start the Dream-Tac server with robot-ready absolute
actions. The normalization mode must match the checkpoint (`q99` for current
checkpoints; use `min_max` only for a checkpoint trained that way).

```bash
cd /path/to/Dream-Tac

export DREAMTAC_CKPT=/path/to/checkpoint
export DREAMTAC_WAN_VAE=/path/to/tokenizer.pth
export DREAMTAC_STATS=/path/to/dataset_statistics_lerobot_bi_flexiv.json
export DREAMTAC_T5=/path/to/t5_embeddings.pkl
export DREAMTAC_DEFAULT_PROMPT='the exact prompt stored in the T5 cache'

python -m cosmos_policy.experiments.robot.bi_flexiv.bi_flexiv_server \
  --host 0.0.0.0 \
  --port 8000 \
  --action-output absolute_from_state \
  --normalization-mode q99 \
  --num-denoising-steps 10
```

The server warms the model before opening the port. Its health endpoint is:

```bash
curl http://127.0.0.1:8000/healthz
```

## Start the robot client

First use the supplied dry-run preset, while selecting the real bench recipe on
the command line:

```bash
cd /path/to/xense-openpi
mamba activate lerobot-xense

python -m examples.dreamtac_bi_flexiv.main \
  --args.run dry-run \
  --args.robot-recipe /path/to/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-05-xgripper.yaml \
  --args.host 192.168.2.100 \
  --args.port 8000
```

Dry-run prevents inferred actions from being sent, but it still connects to the
real robot and episode reset moves both arms to the recipe's start pose. It also
homes/disconnects normally. Treat it as a hardware operation, not a no-motion
configuration.

After checking all seven camera names, the 20D state/action ranges and the
server metadata, start a real synchronous run:

```bash
python -m examples.dreamtac_bi_flexiv.main \
  --args.robot-recipe /path/to/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-05-xgripper.yaml \
  --args.host 192.168.2.100 \
  --args.port 8000 \
  --args.runtime-hz 30 \
  --args.action-hz 0
```

The server default prompt is used unless `--args.prompt` is supplied. A supplied
prompt must be an exact key in the server's T5 embedding cache unless the server
was started with prompt fallback enabled.

Dream-Tac does not support OpenPI RTC. `action_hz=0` is the recommended first
deployment and uses synchronous 30-step chunk execution. After that path is
validated, `--args.action-hz 30` enables the existing decoupled observation and
action runtime; it is pacing, not RTC.

Press Ctrl+C once for graceful shutdown and homing. A second Ctrl+C forces the
process to exit and can leave the arms away from home.
