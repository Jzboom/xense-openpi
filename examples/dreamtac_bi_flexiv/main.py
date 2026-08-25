#!/usr/bin/env python
"""Run Dream-Tac remote inference on the BiFlexiv Rizon4 RT robot."""

from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
import signal
import threading
from typing import override

from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client import action_chunk_broker
from xense_client import paced_broker
from xense_client import websocket_client_policy
from xense_client.runtime import decoupled_runtime
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as synchronous_runtime
from xense_client.runtime.agents import policy_agent

import examples.bi_flexiv_rizon4_rt.recipe as _recipe
from examples.dreamtac_bi_flexiv.env import DreamTacBiFlexivEnvironment
from examples.dreamtac_bi_flexiv.observation import ACTION_DIM
from examples.dreamtac_bi_flexiv.observation import ACTION_HORIZON
from examples.dreamtac_bi_flexiv.policy_adapter import DreamTacRemotePolicy
from examples.dreamtac_bi_flexiv.policy_adapter import validate_server_metadata
from examples.dreamtac_bi_flexiv.robot_config import validate_dreamtac_robot_config
import examples.run_config as _run_config

logger = get_logger("DreamTacBiFlexivMain")

# Run YAMLs shipped with this example; --args.run resolves bare names here.
RUNS_DIR = pathlib.Path(__file__).parent / "runs"


class DryRunEnvironment(_environment.Environment):
    """Connect and observe normally, but never send model actions to the robot."""

    def __init__(self, wrapped: DreamTacBiFlexivEnvironment) -> None:
        self._wrapped = wrapped
        self._step = 0

    @override
    def reset(self) -> None:
        self._wrapped.reset()
        self._step = 0

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return self._wrapped.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        actions = np.asarray(action.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_DIM,):
            raise ValueError(f"Dry-run action must have shape ({ACTION_DIM},), got {actions.shape}")
        self._step += 1
        logger.info(
            f"DRY RUN step {self._step}: "
            f"action range=[{float(actions.min()):+.5f}, {float(actions.max()):+.5f}], "
            f"grips=({float(actions[18]):.4f}, {float(actions[19]):.4f}); not sent"
        )

    def disconnect(self) -> None:
        self._wrapped.disconnect()


@dataclass
class Args:
    """Dream-Tac BiFlexiv robot-client settings."""

    # Optional run YAML under runs/. CLI flags override values from the file.
    run: str | None = None

    # Physical bench. A bare name resolves against the BiFlexiv recipes in this
    # repository; a path may point at a current lerobot-xense recipe. Required:
    # choosing a silent default could connect to the wrong arms.
    robot_recipe: str | None = None

    # Dream-Tac server.
    host: str = "localhost"
    port: int = 8000
    prompt: str | None = None

    # Robot run tuning. Bench hardware and tactile-camera wiring live in the
    # recipe. Force control is pinned off because Dream-Tac emits 20D actions
    # without wrench targets.
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    log_level: str = "INFO"

    # Observation/action scheduling.
    runtime_hz: float = 30.0
    action_hz: float = 0.0
    paced_queue_size: int = 50
    num_episodes: int = 1
    max_episode_steps: int = 1_000_000

    # Safety/debugging. This still connects and resets the real robot.
    dry_run: bool = False

    # Raw images remain local and are never included in the policy RPC.
    include_raw_images: bool = False


def main(args: Args) -> None:
    logger.info(_run_config.describe(args, Args, RUNS_DIR))
    if args.robot_recipe is None:
        raise SystemExit(
            "No bench selected. Pass --args.robot-recipe <name-or-path>, or use a run file that sets it. "
            f"Bundled BiFlexiv recipes: {', '.join(_recipe.available_recipes())}."
        )
    if args.runtime_hz <= 0:
        raise SystemExit(f"--args.runtime-hz must be positive, got {args.runtime_hz}")
    if args.action_hz < 0:
        raise SystemExit(f"--args.action-hz must be non-negative, got {args.action_hz}")

    # Decode and validate the bench before waiting for the policy server or
    # touching hardware. Current lerobot-xense no longer has bi_mount_type or a
    # top-level enable_tactile_sensors field; all bench hardware comes from the
    # recipe and tactile discovery belongs to its typed gripper block.
    recipe_path = _recipe.resolve_recipe_path(args.robot_recipe)
    robot_config = _recipe.load_robot_config(
        recipe_path,
        use_force=False,
        go_to_start=args.go_to_start,
        stiffness_ratio=args.stiffness_ratio,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        log_level=args.log_level,
    )
    validate_dreamtac_robot_config(robot_config)
    logger.info(
        f"Robot recipe: {recipe_path} "
        f"(left={robot_config.left_robot_sn}, right={robot_config.right_robot_sn}, "
        f"gripper={robot_config.gripper.type})"
    )

    websocket_policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = websocket_policy.get_server_metadata()
    validate_server_metadata(metadata)
    logger.info(f"Connected to Dream-Tac server: {metadata}")

    remote_policy = DreamTacRemotePolicy(websocket_policy, default_prompt=args.prompt)
    chunked_policy = action_chunk_broker.ActionChunkBroker(
        policy=remote_policy,
        action_horizon=ACTION_HORIZON,
    )

    base_environment = DreamTacBiFlexivEnvironment(
        robot_config=robot_config,
        include_raw_images=args.include_raw_images,
        setup_robot=True,
    )
    environment: _environment.Environment
    if args.dry_run:
        logger.warn("DRY RUN enabled: robot connects and resets, but inferred actions are not sent")
        environment = DryRunEnvironment(base_environment)
    else:
        environment = base_environment

    if args.action_hz > 0:
        broker = paced_broker.PacedBroker(
            inner=chunked_policy,
            queue_size=args.paced_queue_size,
            target_hz=args.action_hz,
        )
        runtime = decoupled_runtime.DecoupledRuntime(
            environment=environment,
            broker=broker,
            subscribers=[],
            obs_hz=args.runtime_hz,
            action_hz=args.action_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        runtime = synchronous_runtime.Runtime(
            environment=environment,
            agent=policy_agent.PolicyAgent(policy=chunked_policy),
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    shutdown_started = threading.Event()

    def signal_handler(_sig, _frame) -> None:
        if shutdown_started.is_set():
            logger.warn("Second Ctrl+C: forcing exit; robot may not return home cleanly")
            os._exit(1)
        shutdown_started.set()
        logger.info("Ctrl+C: stopping runtime gracefully")
        runtime.request_stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        try:
            environment.disconnect()
        except Exception as exc:
            logger.warn(f"Error disconnecting Dream-Tac robot environment: {exc}")


if __name__ == "__main__":
    main(_run_config.cli(main, Args, RUNS_DIR))
