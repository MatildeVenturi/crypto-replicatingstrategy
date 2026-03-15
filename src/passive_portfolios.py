from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame, price_col: str = "P") -> pd.DataFrame:
    """
    Add log returns by asset:
        r_t = log(P_t / P_{t-1})

    Required columns:
        - datetime
        - fsym
        - price_col
    """
    out = df.sort_values(["fsym", "datetime"]).copy()
    out["r"] = np.log(out[price_col] / out.groupby("fsym")[price_col].shift(1))
    return out


def compute_passive_equal_weight(
    df_panel: pd.DataFrame,
    price_col: str = "P",
) -> pd.DataFrame:
    """
    Equal-weight passive benchmark.

    Logic:
    - all assets receive the same constant weight: 1/N
    - no re-ranking, no signals, no market cap
    - portfolio return at time t is the sum of weighted asset log returns

    Required columns in df_panel:
        - datetime
        - fsym
        - price_col (default: P)

    Returns a dataframe indexed by datetime with:
        - R_PASSIVE_EQUAL
        - Cum_R_PASSIVE_EQUAL
        - Equity_PASSIVE_EQUAL
    """
    required = {"datetime", "fsym", price_col}
    missing = required - set(df_panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_panel.sort_values(["datetime", "fsym"]).copy()
    df = add_log_returns(df, price_col=price_col)

    n_assets = df["fsym"].nunique()
    if n_assets == 0:
        raise ValueError("No assets found in df_panel.")

    df["w_equal"] = 1.0 / float(n_assets)
    df["ret_contrib"] = df["w_equal"] * df["r"]

    out = (
        df.groupby("datetime", as_index=True)["ret_contrib"]
        .sum()
        .to_frame("R_PASSIVE_EQUAL")
    )

    out["Cum_R_PASSIVE_EQUAL"] = out["R_PASSIVE_EQUAL"].cumsum()
    out["Equity_PASSIVE_EQUAL"] = np.exp(out["Cum_R_PASSIVE_EQUAL"])

    return out


def to_daily_returns(
    hourly_portfolio: pd.DataFrame,
    return_col: str = "R_PASSIVE_EQUAL",
) -> pd.DataFrame:
    """
    Aggregate hourly log returns into daily log returns.
    """
    out = hourly_portfolio[[return_col]].dropna().copy()

    daily = out.resample("D").sum()
    daily.columns = [f"{return_col}_daily"]
    daily[f"Cum_{return_col}_daily"] = daily.iloc[:, 0].cumsum()
    daily[f"Equity_{return_col}_daily"] = np.exp(daily[f"Cum_{return_col}_daily"])

    return daily