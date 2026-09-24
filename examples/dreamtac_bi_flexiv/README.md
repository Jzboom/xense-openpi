# Dream-Tac on BiFlexiv Rizon4 RT

This is the robot-side half of the Dream-Tac deployment. The robot computer
does not load OpenPI or Dream-Tac model weights: LeRobot-Xense owns the arms,
grippers and seven cameras, while `xense-client` talks to a Dream-Tac WebSocket
server on a separate GPU computer.

## Data path

```text
LeRobot-Xense recipe -> BiFlexivRizon4RTConfig
  current 20D state + seven synchronized camera frames
        |
        v
DreamTacBiFlexivEnvironment
  RGB history [-3,-2,-1,0] (recent temporal context)
  tactile pair merge + history [-3,-2,-1,0] + consecutive-frame gate
        |
        v
DreamTacRemotePolicy -> Dream-Tac WebSocket server
        |
        v
(40, 20) absolute action chunk -> robot.send_action()
```

The state/action order is:

```text
[left_tcp(0:9), right_tcp(9:18), left_gripper(18), right_gripper(19)]
```

The robot environment captures seven views: `head`, both wrists and two tactile
cameras per gripper. Before transmission, each gripper's raw tactile pair is
stacked vertically, resized to `224x196`, and padded horizontally to `224x224`.
The RPC therefore carries only five condition views: three RGB and
`left_tactile_merged`/`right_tactile_merged`. `images_raw` is never sent to the inference server; when
`--args.include-raw-images` is enabled it stays in the local observation.
Each request carries four chronological frames per camera. All five views use
the recent offsets `[-3,-2,-1,0]`. At episode start, unavailable history is padded by repeating
the first frame, matching training. All five transmitted histories have shape
`(4,224,224,3)`, reducing the uncompressed request from about 15.25 MB to
3.01 MB. The server retains the training-time deterministic center crop.

## Robot recipe

This bench uses
`/home/xense-sn0/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-04-xgripper.yaml`.
The recipe defines both arms, grippers, cameras and start/home poses.
Dream-Tac uses a 20D action and disables force control.

## Start the live RTC policy

On the robot computer (`192.168.204.236`), use the `lerobot-xense` Conda
environment. The Dream-Tac inference server at `192.168.204.183:8000` must
already be running with the 40k checkpoint. These are the only shell commands
needed on the robot computer; the first selects the `jhn` checkout's client
package. The second **sends predicted actions to the real robot**.

```bash
export PYTHONPATH=/home/xense-sn0/jhn/xense-openpi:/home/xense-sn0/jhn/xense-openpi/packages/xense-client/src${PYTHONPATH:+:$PYTHONPATH}
python -m examples.dreamtac_bi_flexiv.main \
  --args.rtc \
  --args.robot-recipe /home/xense-sn0/lerobot-xense/recipes/teleop/bi_flexiv_rizon4_rt/forward-04-xgripper.yaml \
  --args.host 192.168.204.183 \
  --args.port 8000 \
  --args.runtime-hz 30 \
  --args.rtc-prefix 14 \
  --args.rtc-delay-margin 2
```

This runs one episode with the default step limit of 1,000,000; press Ctrl+C
once to stop gracefully and return to the home pose. RTC limits newly
generated targets at 30 Hz to 8 mm of TCP translation, 2 degrees of TCP
rotation, and 0.1 gripper units per step. It also limits the change in TCP
translation to 3 mm/step² at chunk handover and brakes before stationary
targets; frozen prefix actions stay unchanged. The
`RTC generated suffix smoothing` log shows how often the limits are applied.

Robot-side session logs are saved under `examples/dreamtac_bi_flexiv/logs/`
inside this `jhn` checkout. `Step` and `Observed Step` record commanded and
measured poses for comparison; the launch command is unchanged.
