from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import AkShareAshareAdapter, CSVPanelAdapter
from .pipeline import DemoPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factorforge",
        description="LLM-assisted quantitative factor research MVP",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run the offline simulated-data demo")
    demo.add_argument("--config", type=Path, default=None)
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    demo.add_argument(
        "--llm-mode", choices=["mock", "deepseek", "dashscope"], default=None
    )

    fetch = commands.add_parser(
        "fetch", help="optionally download A-share daily data through AkShare"
    )
    fetch.add_argument("--symbols", required=True, help="comma-separated stock codes")
    fetch.add_argument("--start", required=True, help="YYYY-MM-DD")
    fetch.add_argument("--end", required=True, help="YYYY-MM-DD")
    fetch.add_argument("--adjust", choices=["", "qfq", "hfq"], default="qfq")
    fetch.add_argument("--output", type=Path, required=True)

    research = commands.add_parser(
        "research-csv",
        help="run the full benchmark on a user-supplied OHLCV panel CSV",
    )
    research.add_argument("--input", type=Path, required=True)
    research.add_argument("--config", type=Path, default=None)
    research.add_argument("--output-dir", type=Path, required=True)
    research.add_argument(
        "--llm-mode", choices=["mock", "deepseek", "dashscope"], default=None
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        config = load_config(args.config)
        results, report = DemoPipeline(config).run(
            output_dir=args.output_dir, llm_mode=args.llm_mode
        )
        compact = {
            "status": "ok",
            **report["metadata"],
            "experiments": len(results),
            "comparison": str(report["paths"]["comparison"]),
            "summary": str(report["paths"]["summary"]),
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return 0

    if args.command == "research-csv":
        config = load_config(args.config)
        data = CSVPanelAdapter(args.input).load()
        is_simulated = bool(
            "is_simulated" in data.columns and data["is_simulated"].all()
        )
        results, report = DemoPipeline(config).run_data(
            data,
            output_dir=args.output_dir,
            llm_mode=args.llm_mode,
            data_source=(
                "USER_CSV_SIMULATED" if is_simulated else "USER_CSV_REAL_OR_UNVERIFIED"
            ),
            is_simulated=is_simulated,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    **report["metadata"],
                    "experiments": len(results),
                    "summary": str(report["paths"]["summary"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    data = AkShareAshareAdapter(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
    ).load()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.reset_index().to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved {len(data)} real-market rows to {args.output}")
    return 0
