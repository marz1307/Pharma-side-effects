"""Command line entry points for training and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from pharma_side_effects import __version__
from pharma_side_effects.pipelines.predict import predict_from_csv
from pharma_side_effects.pipelines.train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pharma-side-effects",
        description=("Train and serve a side-effect classifier for pharmaceutical metadata."),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train and persist a classifier.")
    train_p.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    train_p.add_argument(
        "--model",
        choices=["random_forest", "logistic_regression"],
        default=None,
        help="Override the configured default model.",
    )
    train_p.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing artifacts to disk.",
    )

    pred_p = sub.add_parser("predict", help="Predict side-effect classes for a CSV.")
    pred_p.add_argument("--model-path", type=Path, required=True)
    pred_p.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    pred_p.add_argument("--output", type=Path, default=None, help="Output CSV path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument list (used by tests).

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        result = train(
            config_path=args.config,
            model_name=args.model,
            persist=not args.no_persist,
        )
        m = result["metrics"]
        print(
            f"model={result['model_name']} accuracy={m['accuracy']:.4f} "
            f"macro_f1={m['macro_f1']:.4f} weighted_f1={m['weighted_f1']:.4f}"
        )
        if "model_path" in result:
            print(f"saved: {result['model_path']}")
        return 0

    if args.command == "predict":
        df = predict_from_csv(args.model_path, args.input, args.output)
        print(f"rows={len(df)} predictions written" if args.output else df.head().to_string())
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
