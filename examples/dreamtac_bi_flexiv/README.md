# Dream-Tac BiFlexiv robot adapter

This example is the robot-side half of the Dream-Tac deployment. It reuses
LeRobot-Xense for hardware access and `xense-client` for WebSocket transport,
runtime scheduling, and action chunk consumption. It does not load OpenPI or a
Dream-Tac model on the robot computer.

## Data path

```text
LeRobot-Xense
  20D state + head/wrists + four tactile frames
        |
        v
DreamTacBiFlexivEnvironment
  direct 224x224 resize + consecutive-frame tactile gate
        |
        v
DreamTacRemotePolicy
  sends only state/images/gate/prompt/observation_seq
        |
        v
Dream-Tac WebSocket server
  returns a (20, 20) absolute action chunk
        |
        v
ActionChunkBroker(horizon=20) -> robot.send_action()
```

`images_raw` is never included in the inference RPC. When
`--include_raw_images` is enabled it remains in the local observation only, for
a future/local recorder or subscriber.

## Start the inference server

On the GPU computer, start the server added to Dream-Tac with
`--action-output absolute_from_state` (the default):

```bash
cd /home/jz/code/dream-tac/Dream-Tac

python -m cosmos_policy.experiments.robot.earbud.earbud_server \
  --host 0.0.0.0 \
  --port 8000 \
  --checkpoint /path/to/checkpoint \
  --wan-vae /path/to/tokenizer.pth \
  --stats /path/to/dataset_statistics_lerobot_earbud.json \
  --t5-embeddings /path/to/t5_embeddings.pkl \
  --default-prompt 'the exact prompt used during training'
```

## Start the robot adapter

Use an environment where this repository's `xense-client` package and the
LeRobot-Xense BiFlexiv implementation are installed:

```bash
python -m pip install -e /home/jz/code/lerobot-xense
python -m pip install -e /home/jz/code/xense-openpi/packages/xense-client
python -m pip install tyro dm-env
```

Verify that `lerobot` resolves to `/home/jz/code/lerobot-xense`, rather than a
different editable fork, before connecting hardware.

```bash
cd /home/jz/code/xense-openpi

python -m examples.dreamtac_bi_flexiv.main \
  --host 192.168.2.100 \
  --port 8000 \
  --runtime_hz 30
```

The server's default prompt is used unless `--prompt` is supplied. The client
validates the metadata handshake and refuses OpenPI servers, non-20-step
policies, RTC policies, or Dream-Tac servers returning temporal-delta actions.

For a first hardware check:

```bash
python -m examples.dreamtac_bi_flexiv.main \
  --host 192.168.2.100 \
  --port 8000 \
  --dry_run \
  --max_episode_steps 20
```

Dry-run still connects to and resets the real robot; it only prevents inferred
actions from being sent. OpenPI RTC is intentionally not exposed. Synchronous
execution is the default. Set `--action_hz 30` to use the existing decoupled
observation/action runtime after the synchronous path has been validated.
