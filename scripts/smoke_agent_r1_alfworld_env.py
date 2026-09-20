#!/usr/bin/env python3
"""Exercise one complete Agent-R1 ALFWorld environment episode without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--game-relative-path", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--agent-r1-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    game_file = (args.data_root / "games" / args.game_relative_path).resolve()
    games_root = (args.data_root / "games").resolve()
    if (
        args.max_steps < 1
        or args.output.exists()
        or not game_file.is_relative_to(games_root)
        or not game_file.is_file()
        or len(args.agent_r1_revision) != 40
    ):
        raise SystemExit("invalid Agent-R1 environment smoke input or output")

    from recipes.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv

    environment = AlfworldTextworldEnv(data_root=str(args.data_root), max_episode_steps=args.max_steps)
    observation, info = environment.reset_with_info(args.game_relative_path)
    trajectory = []
    try:
        for index in range(args.max_steps):
            actions = info.get("admissible_commands", [])
            if not isinstance(actions, list) or not actions:
                raise ValueError(f"no admissible actions at step {index}")
            action = next((item for item in actions if item != "help"), actions[0])
            next_observation, reward, done, next_info = environment.step(action)
            trajectory.append(
                {
                    "step": index + 1,
                    "observation_sha256": hashlib.sha256(observation.encode()).hexdigest(),
                    "admissible_count": len(actions),
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "won": bool(next_info.get("won", False)),
                    "next_observation_sha256": hashlib.sha256(next_observation.encode()).hexdigest(),
                }
            )
            observation, info = next_observation, next_info
            if done:
                break
    finally:
        environment._close_env()

    payload = {
        "artifact": "nonconfirmatory_agent_r1_scripted_environment_episode",
        "model_used": False,
        "training_performed": False,
        "agent_r1_revision": args.agent_r1_revision,
        "game_relative_path": args.game_relative_path,
        "game_sha256": _sha256(game_file),
        "max_steps": args.max_steps,
        "completed_by": "environment_done" if trajectory[-1]["done"] else "step_limit",
        "steps": trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"Agent-R1 ALFWorld scripted episode: {len(trajectory)} steps, "
        f"{payload['completed_by']}, won={trajectory[-1]['won']}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
