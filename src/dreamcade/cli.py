"""``dreamcade`` command line: play, generate data, train, evaluate and export.

ML subcommands import PyTorch lazily so ``dreamcade play`` works from the tiny base install.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

GAMES = ("dungeon", "breakout")


def _require_ml() -> None:
    if importlib.util.find_spec("torch") is None:
        msg = (
            "this command needs the ML extra: run `uv sync --extra ml` (or `uv run --extra ml dreamcade ...`)"
        )
        raise SystemExit(msg)


def _cmd_play(args: argparse.Namespace) -> None:
    from dreamcade.serve import serve

    serve(Path(args.web), args.host, args.port)


def _cmd_gen(args: argparse.Namespace) -> None:
    from dreamcade.config import load_gen
    from dreamcade.data.generate import generate

    spec = load_gen(Path(args.config))
    generate(spec, Path(args.out), args.workers)
    print(f"wrote {spec.shards} {spec.game} shards to {args.out}")


def _cmd_world(args: argparse.Namespace) -> None:
    from dreamcade.dungeon.world import world_to_json

    Path(args.out).write_text(world_to_json(), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


def _cmd_golden(args: argparse.Namespace) -> None:
    from dreamcade.golden import write_golden

    write_golden(Path(args.out), args.steps, args.seed)
    print(f"wrote golden trajectories to {args.out}")


def _cmd_train(args: argparse.Namespace) -> None:
    _require_ml()
    if args.game == "dungeon":
        from dreamcade.train.dungeon import main as train_dungeon

        train_dungeon(Path(args.config), resume=not args.fresh)
    else:
        from dreamcade.train.breakout import main as train_breakout

        train_breakout(Path(args.config), resume=not args.fresh)


def _cmd_eval(args: argparse.Namespace) -> None:
    _require_ml()
    if args.game == "dungeon":
        from dreamcade.evaluate.dungeon import main as eval_dungeon

        eval_dungeon(Path(args.run), Path(args.data), Path(args.out))
    else:
        from dreamcade.evaluate.breakout import main as eval_breakout

        eval_breakout([Path(r) for r in args.runs], Path(args.data), Path(args.out))


def _cmd_export(args: argparse.Namespace) -> None:
    _require_ml()
    from dreamcade.export import export

    export(args.game, Path(args.run), Path(args.out))


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (exposed for tests)."""
    p = argparse.ArgumentParser(prog="dreamcade", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="serve the browser player")
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=8095)
    play.add_argument("--web", default=str(Path(__file__).resolve().parents[2] / "web"))
    play.set_defaults(func=_cmd_play)

    gen = sub.add_parser("gen", help="write episode shards with the exploration policies")
    gen.add_argument("--config", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument("--workers", type=int, default=4)
    gen.set_defaults(func=_cmd_gen)

    world = sub.add_parser("world", help="export the dungeon world for the browser")
    world.add_argument("--out", default="web/assets/dungeon_world.json")
    world.set_defaults(func=_cmd_world)

    golden = sub.add_parser("golden", help="write golden trajectories for the JS parity tests")
    golden.add_argument("--out", required=True)
    golden.add_argument("--steps", type=int, default=300)
    golden.add_argument("--seed", type=int, default=7)
    golden.set_defaults(func=_cmd_golden)

    train = sub.add_parser("train", help="train a world model")
    train.add_argument("game", choices=GAMES)
    train.add_argument("--config", required=True)
    train.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    train.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="evaluate trained models")
    ev.add_argument("game", choices=GAMES)
    ev.add_argument("--run", help="dungeon run directory")
    ev.add_argument("--runs", nargs="*", default=[], help="breakout run directories to compare")
    ev.add_argument("--data", required=True)
    ev.add_argument("--out", default="docs/results")
    ev.set_defaults(func=_cmd_eval)

    exp = sub.add_parser("export", help="export a trained model to ONNX for the browser")
    exp.add_argument("game", choices=GAMES)
    exp.add_argument("--run", required=True)
    exp.add_argument("--out", default="web/models")
    exp.set_defaults(func=_cmd_export)
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
