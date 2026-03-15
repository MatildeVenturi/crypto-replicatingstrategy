from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame, price_col: str = "P") -> pd.DataFrame:
    df = df.sort_values(["fsym", "datetime"]).copy()
    df["r"] = np.log(df[price_col] / df.groupby("fsym")[price_col].shift(1))
    return df


def compute_ts_portfolio(df_with_signal: pd.DataFrame) -> pd.DataFrame:
    """
    Signal-based time-series strategy:
    invest in all N cryptos with weight = Signal_{t-1} / N
    """
    df = df_with_signal.sort_values(["datetime", "fsym"]).copy()
    df = add_log_returns(df, price_col="P")

    n_assets = df["fsym"].nunique()
    df["w_ts"] = df.groupby("fsym")["Signal"].shift(1) / float(n_assets)
    df["ret_contrib"] = df["w_ts"] * df["r"]

    ts = df.groupby("datetime", as_index=True)["ret_contrib"].sum().to_frame("R_TS")
    ts["Cum_R_TS"] = ts["R_TS"].cumsum()
    ts["Equity_TS"] = np.exp(ts["Cum_R_TS"])
    return ts


def _cs_weights_from_score(g: pd.DataFrame, score_col: str) -> pd.DataFrame:
    g = g.copy()
    g["w_cs"] = 0.0

    valid = g[score_col].dropna()
    if len(valid) < 6:
        return g

    top_idx = valid.nlargest(3).index
    bot_idx = valid.nsmallest(3).index

    # paper: +1/6 on each of top 3, -1/6 on each of bottom 3
    g.loc[top_idx, "w_cs"] = 1.0 / 6.0
    g.loc[bot_idx, "w_cs"] = -1.0 / 6.0
    return g


def compute_cs_portfolio(df_with_signal: pd.DataFrame) -> pd.DataFrame:
    """
    Signal-based cross-sectional strategy:
    rank by Signal_t, but apply weights at t-1 to returns at t.
    """
    df = df_with_signal.sort_values(["datetime", "fsym"]).copy()
    df = add_log_returns(df, price_col="P")

    df = (
        df.groupby("datetime", group_keys=False)
        .apply(lambda g: _cs_weights_from_score(g, score_col="Signal"))
        .reset_index(drop=True)
    )

    df["w_cs_lag"] = df.groupby("fsym")["w_cs"].shift(1)
    df["ret_contrib"] = df["w_cs_lag"] * df["r"]

    cs = df.groupby("datetime", as_index=True)["ret_contrib"].sum().to_frame("R_CS")
    cs["Cum_R_CS"] = cs["R_CS"].cumsum()
    cs["Equity_CS"] = np.exp(cs["Cum_R_CS"])
    return cs


def add_returns_based_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previous-period return used for returns-based strategies.
    """
    df = df.sort_values(["fsym", "datetime"]).copy()
    df = add_log_returns(df, price_col="P")
    df["ret_prev"] = df.groupby("fsym")["r"].shift(1)
    return df


def compute_ts_portfolio_returns_based(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Returns-based time-series strategy:
    if previous return > 0 => +1/N
    if previous return < 0 => -1/N
    else 0
    """
    df = add_returns_based_signal(df_panel)
    n_assets = df["fsym"].nunique()

    df["sgn_prev_ret"] = np.sign(df["ret_prev"]).fillna(0.0)
    df["w_ts_rb"] = df["sgn_prev_ret"] / float(n_assets)
    df["ret_contrib"] = df["w_ts_rb"] * df["r"]

    ts = df.groupby("datetime", as_index=True)["ret_contrib"].sum().to_frame("R_TS_RB")
    ts["Cum_R_TS_RB"] = ts["R_TS_RB"].cumsum()
    ts["Equity_TS_RB"] = np.exp(ts["Cum_R_TS_RB"])
    return ts


def compute_cs_portfolio_returns_based(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Returns-based cross-sectional strategy:
    rank by previous return, long top 3 (+1/6), short bottom 3 (-1/6)
    """
    df = add_returns_based_signal(df_panel)

    df = (
        df.groupby("datetime", group_keys=False)
        .apply(lambda g: _cs_weights_from_score(g, score_col="ret_prev"))
        .reset_index(drop=True)
    )

    df["ret_contrib"] = df["w_cs"] * df["r"]

    cs = df.groupby("datetime", as_index=True)["ret_contrib"].sum().to_frame("R_CS_RB")
    cs["Cum_R_CS_RB"] = cs["R_CS_RB"].cumsum()
    cs["Equity_CS_RB"] = np.exp(cs["Cum_R_CS_RB"])
    return cs


def to_daily_returns(hourly_portfolio: pd.DataFrame, return_col: str) -> pd.DataFrame:
    out = hourly_portfolio.copy()
    out = out[[return_col]].dropna()
    daily = out.resample("D").sum()
    daily.columns = [f"{return_col}_daily"]
    daily[f"Cum_{return_col}_daily"] = daily.iloc[:, 0].cumsum()
    daily[f"Equity_{return_col}_daily"] = np.exp(daily[f"Cum_{return_col}_daily"])
    return daily


def summary_stats(returns: pd.Series, periods_per_year: float = 24.0 * 365.0) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {
            "n_obs": 0,
            "mean": np.nan,
            "std": np.nan,
            "ann_mean": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "cum_log_return": np.nan,
            "cum_simple_return": np.nan,
            "max_drawdown": np.nan,
        }

    mean_r = returns.mean()
    std_r = returns.std(ddof=1)

    cum_log = returns.sum()
    equity = np.exp(returns.cumsum())
    peak = equity.cummax()
    dd = equity / peak - 1.0

    ann_mean = mean_r * periods_per_year
    ann_vol = std_r * np.sqrt(periods_per_year) if std_r > 0 else np.nan
    sharpe = ann_mean / ann_vol if (pd.notna(ann_vol) and ann_vol > 0) else np.nan

    return {
        "n_obs": int(returns.shape[0]),
        "mean": float(mean_r),
        "std": float(std_r),
        "ann_mean": float(ann_mean),
        "ann_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "cum_log_return": float(cum_log),
        "cum_simple_return": float(np.exp(cum_log) - 1.0),
        "max_drawdown": float(dd.min()),
    }