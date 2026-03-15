from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame, price_col: str = "P") -> pd.DataFrame:
    df = df.sort_values(["fsym", "datetime"]).copy()
    df["r"] = np.log(df[price_col] / df.groupby("fsym")[price_col].shift(1))
    return df


def prepare_market_cap_panel(
    mcap_df: pd.DataFrame,
    datetime_col: str = "datetime",
    symbol_col: str = "fsym",
    mcap_col: str = "market_cap_usd",
) -> pd.DataFrame:
    """
    Input expected:
        datetime, fsym, market_cap_usd

    Returns:
        clean panel with monthly weight snapshots computed from the
        first available timestamp of each month for each symbol.
    """
    df = mcap_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
    df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
    df[symbol_col] = df[symbol_col].astype(str).str.upper()

    df = df.dropna(subset=[datetime_col, symbol_col, mcap_col]).copy()
    df = df[df[mcap_col] > 0].copy()

    df = df.rename(
        columns={
            datetime_col: "datetime",
            symbol_col: "fsym",
            mcap_col: "market_cap_usd",
        }
    )

    df["month"] = df["datetime"].dt.to_period("M")

    # first available observation in each month for each symbol
    df = df.sort_values(["fsym", "datetime"])
    first_month_obs = (
        df.groupby(["fsym", "month"], as_index=False)
        .first()
        .copy()
    )

    # normalize into weights by month
    first_month_obs["month_total_mcap"] = first_month_obs.groupby("month")["market_cap_usd"].transform("sum")
    first_month_obs["weight_monthly"] = first_month_obs["market_cap_usd"] / first_month_obs["month_total_mcap"]

    return first_month_obs[["month", "fsym", "datetime", "market_cap_usd", "weight_monthly"]].copy()


def compute_passive_fixed_weights(
    df_panel: pd.DataFrame,
    weight_datetime: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Passive portfolio with fixed weights:
    weights are computed once from indexed prices / proxy market value at the chosen start date.

    IMPORTANT:
    this is only a fallback if you do NOT have market cap data.
    It uses relative price levels at the start date as a proxy, which is NOT identical to the paper.
    The paper uses market-cap-based weights. Use compute_passive_fixed_weights_from_mcap if possible.
    """
    df = df_panel.sort_values(["datetime", "fsym"]).copy()
    df = add_log_returns(df, price_col="P")

    if weight_datetime is None:
        weight_datetime = df["datetime"].min()

    w0 = df[df["datetime"] == weight_datetime][["fsym", "P"]].copy()
    w0 = w0.dropna()
    if w0.empty:
        raise ValueError("No observations found at the chosen weight_datetime.")

    w0["w_fixed"] = w0["P"] / w0["P"].sum()

    df = df.merge(w0[["fsym", "w_fixed"]], on="fsym", how="left")
    df["ret_contrib"] = df["w_fixed"] * df["r"]

    out = df.groupby("datetime", as_index=True)["ret_contrib"].sum().to_frame("R_PASSIVE_FIXED")
    out["Cum_R_PASSIVE_FIXED"] = out["R_PASSIVE_FIXED"].cumsum()
    out["Equity_PASSIVE_FIXED"] = np.exp(out["Cum_R_PASSIVE_FIXED"])
    return out


def compute_passive_fixed_weights_from_mcap(
    df_panel: pd.DataFrame,
    mcap_monthly_weights: pd.DataFrame,
    start_month: str | None = None,
) -> pd.DataFrame:
    """
    Paper-like fixed passive portfolio:
    weights computed from market cap shares on the first implementation month,
    then kept fixed over the whole sample.
    """
    df = df_panel.sort_values(["datetime", "fsym"]).copy()
    df = add_log_returns(df, price_col="P")

    weights = mcap_monthly_weights.copy()
    weights["month"] = weights["month"].astype(str)

    if start_month is None:
        start_month = str(df["datetime"].min().to_period("M"))

    w0 = weights[weights["month"] == start_month][["fsym", "weight_monthly"]].copy()
    if w0.empty:
        raise ValueError(f"No market-cap weights found for start month {start_month}.")

    w0 = w0.rename(columns={"weight_monthly": "w_fixed"})
    df = df.merge(w0, on="fsym", how="left")

    df["ret_contrib"] = df["w_fixed"] * df["r"]

    out = df.groupby("datetime", as_index=True)["ret_contrib"].sum().to_frame("R_PASSIVE_FIXED")
    out["Cum_R_PASSIVE_FIXED"] = out["R_PASSIVE_FIXED"].cumsum()
    out["Equity_PASSIVE_FIXED"] = np.exp(out["Cum_R_PASSIVE_FIXED"])
    return out


def compute_passive_monthly_varying_weights_from_mcap(
    df_panel: pd.DataFrame,
    mcap_monthly_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paper-like monthly varying passive portfolio:
    each month uses market-cap shares from the first day / first available timestamp of that month.
    """
    df = df_panel.sort_values(["datetime", "fsym"]).copy()
    df = add_log_returns(df, price_col="P")
    df["month"] = df["datetime"].dt.to_period("M").astype(str)

    weights = mcap_monthly_weights.copy()
    weights["month"] = weights["month"].astype(str)

    df = df.merge(
        weights[["month", "fsym", "weight_monthly"]],
        on=["month", "fsym"],
        how="left",
    )

    df["ret_contrib"] = df["weight_monthly"] * df["r"]

    out = df.groupby("datetime", as_index=True)["ret_contrib"].sum().to_frame("R_PASSIVE_MONTHLY")
    out["Cum_R_PASSIVE_MONTHLY"] = out["R_PASSIVE_MONTHLY"].cumsum()
    out["Equity_PASSIVE_MONTHLY"] = np.exp(out["Cum_R_PASSIVE_MONTHLY"])
    return out


def to_daily_returns(hourly_portfolio: pd.DataFrame, return_col: str) -> pd.DataFrame:
    out = hourly_portfolio[[return_col]].dropna().copy()
    daily = out.resample("D").sum()
    daily.columns = [f"{return_col}_daily"]
    daily[f"Cum_{return_col}_daily"] = daily.iloc[:, 0].cumsum()
    daily[f"Equity_{return_col}_daily"] = np.exp(daily[f"Cum_{return_col}_daily"])
    return daily