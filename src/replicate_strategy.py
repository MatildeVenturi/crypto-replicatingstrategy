# src/replicate_strategy.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHORT_N = (8, 16, 32)
LONG_N  = (24, 48, 96)


# -----------------------------
# 1) Data loading + cleaning
# -----------------------------
def load_hourly_crypto_csv(
    csv_path: str,
    symbol: str | None = "BTC",
    datetime_col: str = "datetime",
    price_col: str = "price_usd",
    symbol_col_candidates: tuple[str, ...] = ("fsvm", "fsym", "symbol", "coin"),
) -> pd.DataFrame:
    """
    Loads a CryptoCompare-like hourly csv and returns a clean hourly time series:
      - parsed UTC datetime
      - numeric price
      - optional symbol filter (only if a symbol column exists)
      - sorted, de-duplicated timestamps
      - regular hourly grid with time interpolation for small gaps

    Returns a DataFrame with index = hourly UTC DatetimeIndex and a 'price_usd' column.
    """
    df = pd.read_csv(csv_path)

    # Find symbol column (if any)
    sym_col = None
    for c in symbol_col_candidates:
        if c in df.columns:
            sym_col = c
            break

    # Parse datetime + price
    if datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column '{datetime_col}'. Columns: {df.columns.tolist()}")

    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}'. Columns: {df.columns.tolist()}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=[datetime_col, price_col]).copy()
    df = df[df[price_col] > 0].copy()

    # Optional symbol filter
    if symbol is not None and sym_col is not None:
        df = df[df[sym_col].astype(str).str.upper() == symbol.upper()].copy()

    # Sort + deduplicate timestamps
    df = df.sort_values(datetime_col)
    df = df.drop_duplicates(subset=[datetime_col], keep="last")

    # Make hourly grid
    df = df.set_index(datetime_col)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_idx)

    # Interpolate missing prices on time, then ffill/bfill as last resort
    df[price_col] = df[price_col].interpolate("time").ffill().bfill()

    # Keep only what we need
    out = pd.DataFrame({price_col: df[price_col]})
    out.index.name = "datetime"
    return out


# -----------------------------
# 2) EMA + strategy replication
# -----------------------------
def ema(series: pd.Series, n: int) -> pd.Series:
    """
    Paper recursion: EMA_t = alpha * P_t + (1-alpha) * EMA_{t-1}, alpha = 1/n
    pandas ewm(adjust=False) matches that recursion.
    """
    alpha = 1.0 / float(n)
    return series.ewm(alpha=alpha, adjust=False).mean()


def replicate_momentum_strategy(
    df: pd.DataFrame,
    price_col: str = "price_usd",
    short_list: tuple[int, ...] = SHORT_N,
    long_list: tuple[int, ...] = LONG_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replicates EMA-crossover momentum:

      signal_t = +1 if EMA_short > EMA_long else -1
      position_t = signal_{t-1}     (shift to avoid look-ahead)
      strat_ret_t = position_t * ret_t

    IMPORTANT:
      EMAs are computed on *raw price* (scale-invariant for crossover).
      Returns are computed on raw price.

    Returns:
      - full dataframe with EMAs, signals, positions, returns
      - summary table for all (short,long) pairs
    """
    out = df.copy()

    # Returns (hourly)
    out["ret"] = out[price_col].pct_change().fillna(0.0)

    # Precompute all required EMAs
    all_n = sorted(set(short_list) | set(long_list))
    for n in all_n:
        out[f"ema_{n}"] = ema(out[price_col], n)

    rows = []
    ann_factor = 24.0 * 365.0

    for ns in short_list:
        for nl in long_list:
            sig = np.where(out[f"ema_{ns}"] > out[f"ema_{nl}"], 1.0, -1.0)
            pos = pd.Series(sig, index=out.index).shift(1).fillna(0.0)
            strat_ret = pos * out["ret"]

            equity = (1.0 + strat_ret).cumprod()
            total_return = float(equity.iloc[-1] - 1.0)

            mean_r = float(strat_ret.mean())
            std_r = float(strat_ret.std(ddof=1)) if len(strat_ret) > 1 else np.nan

            ann_vol = std_r * np.sqrt(ann_factor) if np.isfinite(std_r) else np.nan
            sharpe = (mean_r / std_r) * np.sqrt(ann_factor) if (np.isfinite(std_r) and std_r > 0) else np.nan
            ann_return_est = (1.0 + mean_r) ** ann_factor - 1.0 if np.isfinite(mean_r) else np.nan

            peak = equity.cummax()
            dd = equity / peak - 1.0
            max_dd = float(dd.min())

            # crude turnover proxy = number of position changes
            pos_changes = int((pos.diff().abs() > 0).sum())

            # store in out
            out[f"signal_s{ns}_l{nl}"] = sig
            out[f"pos_s{ns}_l{nl}"] = pos
            out[f"strat_ret_s{ns}_l{nl}"] = strat_ret
            out[f"equity_s{ns}_l{nl}"] = equity

            rows.append({
                "short_n": ns,
                "long_n": nl,
                "total_return": total_return,
                "ann_return_est": ann_return_est,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "position_changes": pos_changes,
            })

    summary = pd.DataFrame(rows).sort_values(["short_n", "long_n"]).reset_index(drop=True)
    return out, summary


# -----------------------------
# 3) Paper-like plotting
# -----------------------------
def plot_paper_like_window(
    df_full: pd.DataFrame,
    out_path: str,
    price_col: str = "price_usd",
    short_n: int = 8,
    long_n: int = 24,
    start_idx: int = 2000,
    window_hours: int = 1000,
    use_mean_norm_for_plot: bool = False,
):
    """
    Nice paper-like plot:
      - choose a window
      - normalize within window (either by first price (recommended) or by mean (plot-only))
      - compute EMAs on the normalized window (plot-only)
      - x-axis is Hours (0..window-1)
    """
    d = df_full.copy()

    # select window (by row index, since we already have a clean hourly grid)
    start_idx = max(0, int(start_idx))
    end_idx = min(start_idx + int(window_hours), len(d))
    w = d.iloc[start_idx:end_idx].copy()

    if len(w) < 10:
        raise ValueError("Window too short to plot. Reduce start_idx or increase data coverage.")

    # normalize ONLY for plot aesthetics
    if use_mean_norm_for_plot:
        denom = float(w[price_col].mean())
        label_price = "Bitcoin (Mean-normalized)"
    else:
        denom = float(w[price_col].iloc[0])
        label_price = "Bitcoin (Indexed Price)"

    w["P_plot"] = w[price_col] / denom

    # EMAs for plot (scale-invariant; computed on P_plot just for matching the visual scale)
    w["ema_s_plot"] = ema(w["P_plot"], short_n)
    w["ema_l_plot"] = ema(w["P_plot"], long_n)

    x = np.arange(len(w))

    plt.figure(figsize=(8.6, 3.6))  # close to typical paper aspect
    plt.plot(x, w["P_plot"].to_numpy(), linewidth=1.0, label=label_price)
    plt.plot(x, w["ema_s_plot"].to_numpy(), linewidth=1.0, label=f"EMA Short (n={short_n})")
    plt.plot(x, w["ema_l_plot"].to_numpy(), linewidth=1.0, label=f"EMA Long (n={long_n})")

    plt.xlabel("Hours")
    plt.ylabel("Indexed Price" if not use_mean_norm_for_plot else "Mean-normalized Price")

    # light grid like papers
    plt.grid(True, linewidth=0.4, alpha=0.4)

    # tight y-limits
    y = w["P_plot"].to_numpy()
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.03 * (ymax - ymin)
        plt.ylim(ymin - pad, ymax + pad)

    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# 4) Save tables
# -----------------------------
def save_tables(summary: pd.DataFrame, out_dir: str, prefix: str = "strategy_summary"):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    md_path = os.path.join(out_dir, f"{prefix}.md")

    summary.to_csv(csv_path, index=False)

    md = summary.copy()
    for c in ["total_return", "ann_return_est", "ann_vol", "sharpe", "max_drawdown"]:
        md[c] = md[c].astype(float).map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md.to_markdown(index=False))

    print(f"[OK] Tables saved:\n  - {csv_path}\n  - {md_path}")


# -----------------------------
# 5) CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/cryptocompare_hourly_usd_2020_T4141.csv")
    parser.add_argument("--symbol", default="BTC")  # ignored if no symbol column exists
    parser.add_argument("--out_dir", default="tables")

    parser.add_argument("--plot_short", type=int, default=8)
    parser.add_argument("--plot_long", type=int, default=24)
    parser.add_argument("--plot_start", type=int, default=2000)
    parser.add_argument("--plot_window", type=int, default=1000)
    parser.add_argument("--plot_mean_norm", action="store_true")  # plot-only

    args = parser.parse_args()

    # Load clean hourly series
    df = load_hourly_crypto_csv(args.csv, symbol=args.symbol)

    # Replicate strategy
    full, summary = replicate_momentum_strategy(df)

    # Save performance table
    save_tables(summary, args.out_dir, prefix=f"strategy_summary_{args.symbol}")

    # Save paper-like plot
    plot_path = os.path.join(
        args.out_dir,
        f"fig2_like_{args.symbol}_ema{args.plot_short}_{args.plot_long}_start{args.plot_start}_win{args.plot_window}.png"
    )
    plot_paper_like_window(
        full,
        out_path=plot_path,
        short_n=args.plot_short,
        long_n=args.plot_long,
        start_idx=args.plot_start,
        window_hours=args.plot_window,
        use_mean_norm_for_plot=args.plot_mean_norm,
    )
    print(f"[OK] Plot saved:\n  - {plot_path}")


if __name__ == "__main__":
    main()