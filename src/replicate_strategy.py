from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from passive_portfolios import compute_passive_equal_weight, to_daily_returns

from signals_momentum import add_signals
from portfolios import (
    compute_ts_portfolio,
    compute_cs_portfolio,
    compute_ts_portfolio_returns_based,
    compute_cs_portfolio_returns_based,
    to_daily_returns,
    summary_stats,
)

PARAM_GRID = [
    (12, 168),
    (24, 168),
    (12, 720),
    (24, 720),
]


def load_panel_csv(
    csv_path: str,
    datetime_col: str = "datetime",
    price_col: str = "price_usd",
    symbol_col_candidates: tuple[str, ...] = ("fsym", "fsvm", "symbol", "coin"),
    selected_symbols: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    sym_col = None
    for c in symbol_col_candidates:
        if c in df.columns:
            sym_col = c
            break
    if sym_col is None:
        raise ValueError(
            f"Could not find symbol column. Tried: {symbol_col_candidates}. "
            f"Columns are: {df.columns.tolist()}"
        )

    if datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column '{datetime_col}'")
    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}'")

    df = df.rename(columns={sym_col: "fsym", datetime_col: "datetime", price_col: "price_usd"}).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df["fsym"] = df["fsym"].astype(str).str.upper()

    df = df.dropna(subset=["datetime", "fsym", "price_usd"]).copy()
    df = df[df["price_usd"] > 0].copy()

    if selected_symbols is not None and len(selected_symbols) > 0:
        selected = {s.upper() for s in selected_symbols}
        df = df[df["fsym"].isin(selected)].copy()

    df = df.sort_values(["fsym", "datetime"]).drop_duplicates(["fsym", "datetime"], keep="last")

    # regular hourly grid per symbol
    chunks = []
    for sym, g in df.groupby("fsym"):
        g = g.sort_values("datetime").set_index("datetime")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="H", tz="UTC")
        g = g.reindex(full_idx)
        g["price_usd"] = g["price_usd"].interpolate("time").ffill().bfill()
        g["fsym"] = sym
        g = g.reset_index().rename(columns={"index": "datetime"})
        chunks.append(g)

    out = pd.concat(chunks, ignore_index=True)
    out = out[["datetime", "fsym", "price_usd"]].sort_values(["datetime", "fsym"]).reset_index(drop=True)
    return out

#keep only valid entries in dataset
def restrict_common_panel(df: pd.DataFrame) -> pd.DataFrame:
  
    counts = df.groupby("datetime")["fsym"].nunique()
    n_assets = df["fsym"].nunique()
    valid_dt = counts[counts == n_assets].index
    return df[df["datetime"].isin(valid_dt)].copy()


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def save_table(df: pd.DataFrame, path_csv: Path, path_md: Path) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_csv, index=False)

    md = df.copy()
    for c in md.columns:
        if pd.api.types.is_float_dtype(md[c]):
            md[c] = md[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write(md.to_markdown(index=False))


def make_comparison_plot(
    ts: pd.DataFrame,
    cs: pd.DataFrame,
    ts_rb: pd.DataFrame,
    cs_rb: pd.DataFrame,
    passive_eq: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(ts.index, ts["Equity_TS"], label="Signal-based TS")
    plt.plot(cs.index, cs["Equity_CS"], label="Signal-based CS")
    plt.plot(ts_rb.index, ts_rb["Equity_TS_RB"], label="Returns-based TS")
    plt.plot(cs_rb.index, cs_rb["Equity_CS_RB"], label="Returns-based CS")
    plt.plot(passive_eq.index, passive_eq["Equity_PASSIVE_EQUAL"], label="Passive equal-weight")

    plt.title(title)
    plt.xlabel("Datetime")
    plt.ylabel("Equity (exp cumulative log return)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def build_summary_row(name: str, ret_series: pd.Series, t1: int, t2: int) -> dict:
    stats = summary_stats(ret_series)
    stats["strategy"] = name
    stats["t1"] = t1
    stats["t2"] = t2
    return stats


def run_one_parameterization(df_panel: pd.DataFrame, t1: int, t2: int, out_dir: Path) -> pd.DataFrame:
    print(f"[INFO] Running t1={t1}, t2={t2}")

   
    sig_panel = add_signals(df_panel, price_col="price_usd", t1=t1, t2=t2)

   
    sig_panel = sig_panel.dropna(subset=["Signal", "P"]).copy()
    sig_panel = restrict_common_panel(sig_panel)

    # signal-based portfolios
    ts = compute_ts_portfolio(sig_panel)
    cs = compute_cs_portfolio(sig_panel)

    
    base_panel = sig_panel[["datetime", "fsym", "price_usd", "P"]].copy()
    ts_rb = compute_ts_portfolio_returns_based(base_panel)
    cs_rb = compute_cs_portfolio_returns_based(base_panel)

    passive_eq = compute_passive_equal_weight(base_panel)
    passive_eq_d = to_daily_returns(passive_eq, "R_PASSIVE_EQUAL")

    # daily returns
    ts_d = to_daily_returns(ts, "R_TS")
    cs_d = to_daily_returns(cs, "R_CS")
    ts_rb_d = to_daily_returns(ts_rb, "R_TS_RB")
    cs_rb_d = to_daily_returns(cs_rb, "R_CS_RB")

    tag = f"t1_{t1}_t2_{t2}"

    save_df(sig_panel.set_index("datetime"), out_dir / "processed" / f"signals_{tag}.csv")
    save_df(ts, out_dir / "hourly" / f"ts_signal_{tag}.csv")
    save_df(cs, out_dir / "hourly" / f"cs_signal_{tag}.csv")
    save_df(ts_rb, out_dir / "hourly" / f"ts_returns_based_{tag}.csv")
    save_df(cs_rb, out_dir / "hourly" / f"cs_returns_based_{tag}.csv")
    save_df(passive_eq, out_dir / "hourly" / f"passive_equal_{tag}.csv")
    save_df(passive_eq_d, out_dir / "daily" / f"passive_equal_daily_{tag}.csv")

    save_df(ts_d, out_dir / "daily" / f"ts_signal_daily_{tag}.csv")
    save_df(cs_d, out_dir / "daily" / f"cs_signal_daily_{tag}.csv")
    save_df(ts_rb_d, out_dir / "daily" / f"ts_returns_based_daily_{tag}.csv")
    save_df(cs_rb_d, out_dir / "daily" / f"cs_returns_based_daily_{tag}.csv")

    make_comparison_plot(
    ts,
    cs,
    ts_rb,
    cs_rb,
    passive_eq,
    out_path=out_dir / "plots" / f"comparison_{tag}.png",
    title=f"Momentum strategies comparison ({tag})",
)

    rows = [
        build_summary_row("signal_TS", ts["R_TS"], t1, t2),
        build_summary_row("signal_CS", cs["R_CS"], t1, t2),
        build_summary_row("returns_TS", ts_rb["R_TS_RB"], t1, t2),
        build_summary_row("returns_CS", cs_rb["R_CS_RB"], t1, t2),
        build_summary_row("passive_equal", passive_eq["R_PASSIVE_EQUAL"], t1, t2),
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/processed/cryptocompare_hourly_usd_2020_T4141.csv",
        help="Panel CSV with datetime, symbol, price_usd",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional subset of symbols, e.g. BTC ETH DASH LTC MAID XMR XRP",
    )
    parser.add_argument(
        "--out_dir",
        default="reports/backtests",
        help="Output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    df_panel = load_panel_csv(args.csv, selected_symbols=args.symbols)
    df_panel = restrict_common_panel(df_panel)

    all_summaries = []
    for t1, t2 in PARAM_GRID:
        summ = run_one_parameterization(df_panel, t1=t1, t2=t2, out_dir=out_dir)
        all_summaries.append(summ)

    summary = pd.concat(all_summaries, ignore_index=True)
    summary = summary[
        ["strategy", "t1", "t2", "n_obs", "mean", "std", "ann_mean", "ann_vol", "sharpe",
         "cum_log_return", "cum_simple_return", "max_drawdown"]
    ].sort_values(["t1", "t2", "strategy"]).reset_index(drop=True)

    save_table(
        summary,
        out_dir / "tables" / "strategy_summary.csv",
        out_dir / "tables" / "strategy_summary.md",
    )

    print("\n[OK] Finished.")
    print(f"[OK] Summary saved to: {out_dir / 'tables' / 'strategy_summary.csv'}")


if __name__ == "__main__":
    main()