"""Command line entrypoint: ``sentimetro <command> [--config ...] [--set k=v ...]``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from sentimetro.config import Config, load_config
from sentimetro.utils import setup_logging

logger = logging.getLogger("sentimetro")


def _download(cfg: Config, _: argparse.Namespace) -> None:
    from sentimetro.data.download import download_dataset

    download_dataset(cfg.data)


def _preprocess(cfg: Config, _: argparse.Namespace) -> None:
    from sentimetro.data.preprocess import preprocess

    preprocess(cfg.data)


def _train_baseline(cfg: Config, _: argparse.Namespace) -> None:
    from sentimetro.training.baseline import train_baseline

    train_baseline(cfg)


def _train_transformer(cfg: Config, _: argparse.Namespace) -> None:
    from sentimetro.training.transformer import train_transformer

    train_transformer(cfg)


def _evaluate(cfg: Config, _: argparse.Namespace) -> None:
    from sentimetro.evaluation.report import evaluate_all

    evaluate_all(cfg)


def _export(cfg: Config, _: argparse.Namespace) -> None:
    from sentimetro.export import export_production

    export_production(cfg)


def _serve(_: Config, args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "sentimetro.serving.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_config=None,
    )


COMMANDS: dict[str, tuple[Callable[[Config, argparse.Namespace], None], str]] = {
    "download": (_download, "download the raw dataset from the Hugging Face Hub"),
    "preprocess": (_preprocess, "clean, dedupe and split the raw data"),
    "train-baseline": (_train_baseline, "train the TF-IDF + logistic regression baseline"),
    "train-transformer": (_train_transformer, "fine-tune the transformer"),
    "evaluate": (_evaluate, "evaluate the models on the test split and write reports/"),
    "export": (_export, "export the best model to models/production (ONNX)"),
    "serve": (_serve, "run the FastAPI inference service"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentimetro", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set transformer.epochs=3 (repeatable)",
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        cmd = sub.add_parser(name, help=help_text)
        if name == "serve":
            cmd.add_argument("--host", default="127.0.0.1")
            cmd.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(args.config if args.config.exists() else None, args.overrides)
    handler, _ = COMMANDS[args.command]
    handler(cfg, args)


if __name__ == "__main__":
    main()
