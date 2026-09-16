"""
    python tools/widen_checkpoint.py --source agent_code/attackontensor_ppo/policy.pt --output checkpoints/ppo13-s2-widened17.pt --survival-channels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_code.attackontensor_ppo import tensorizer as T  # noqa: E402

FIRST_CONV_WEIGHT = "backbone.0.weight"


def widen(payload: dict, target_channels: int) -> dict:
    """Return a copy of ``payload`` whose first conv accepts ``target_channels``."""
    state = dict(payload["state_dict"])
    weight = state[FIRST_CONV_WEIGHT]
    current = int(weight.shape[1])
    if target_channels < current:
        raise ValueError(f"cannot shrink {current} input planes to {target_channels}")
    if target_channels > current:
        extra = torch.zeros(
            weight.shape[0], target_channels - current, *weight.shape[2:],
            dtype=weight.dtype,
        )
        state[FIRST_CONV_WEIGHT] = torch.cat([weight, extra], dim=1)

    network = dict(payload.get("network") or {})
    network["in_channels"] = target_channels

    config = dict(payload.get("config") or {})
    config["survival_channels"] = target_channels == T.N_CHANNELS

    widened = dict(payload)
    widened.update(state_dict=state, network=network, config=config)
    widened["widened_from_channels"] = current
    return widened


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--survival-channels", action="store_true",
                        help=f"widen to the {T.N_CHANNELS}-plane layout with the "
                             "survival-profile planes")
    args = parser.parse_args(argv)

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    target = T.N_CHANNELS if args.survival_channels else T.BASE_CHANNELS
    widened = widen(payload, target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(widened, args.output)
    before = int(payload["state_dict"][FIRST_CONV_WEIGHT].shape[1])
    print(f"{args.source} ({before} planes) -> {args.output} ({target} planes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
