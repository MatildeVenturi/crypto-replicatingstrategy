from __future__ import annotations

import numpy as np
import pandas as pd

SHORTS = (8, 16, 32)
LONGS = (24, 48, 96)


def ema_alpha(series: pd.Series, n: int) -> pd.Series:
    """
    Paper EMA recursion:
        EMA_t = alpha * P_t + (1-alpha) * EMA_{t-1}
    with alpha = 1/n
    """
    return series.ewm(alpha=1.0 / float(n), adjust=False).mean()


def compute_indexed_price(g: pd.DataFrame, price_col: str = "price_usd") -> pd.Series:
    p0 = g[price_col].iloc[0]
    if pd.isna(p0) or p0 == 0:
        raise ValueError("Initial price is missing or zero; cannot index prices.")
    return g[price_col] / p0


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=numer.index, dtype="float64")
    mask = denom.notna() & numer.notna() & (denom != 0)
    out.loc[mask] = numer.loc[mask] / denom.loc[mask]
    return out


def scale_signal(z: pd.Series) -> pd.Series:
    """
    Stable implementation of the paper scaling:
        u(z) = z * exp(-z^2 / 4) / sqrt(2)
    """
    return z * np.exp(-(z ** 2) / 4.0) / np.sqrt(2.0)


def compute_signal_for_group(
    g: pd.DataFrame,
    price_col: str = "price_usd",
    t1: int = 12,
    t2: int = 168,
) -> pd.DataFrame:
    """
    Computes the paper's momentum signal for one crypto.

    Returns columns:
        datetime, fsym, price_usd, P, Signal
    plus optional intermediate columns.
    """
    g = g.sort_values("datetime").copy()
    g["P"] = compute_indexed_price(g, price_col=price_col)

    sigma_p = g["P"].rolling(window=t1, min_periods=t1).std()

    uk_cols = []

    for k, (ns, nl) in enumerate(zip(SHORTS, LONGS), start=1):
        ema_s = ema_alpha(g["P"], ns)
        ema_l = ema_alpha(g["P"], nl)

        # Eq. (2): x_k = (EMA_short - EMA_long) / P
        xk = (ema_s - ema_l) / g["P"]

        # Eq. (3): y_k = x_k / std_t1(P)
        yk = safe_div(xk, sigma_p)

        # Eq. (4): z_k = y_k / std_t2(y_k)
        sigma_y = yk.rolling(window=t2, min_periods=t2).std()
        zk = safe_div(yk, sigma_y)

        # Eq. (5): scaled signal
        uk = scale_signal(zk)

        g[f"ema_s_{k}"] = ema_s
        g[f"ema_l_{k}"] = ema_l
        g[f"x_{k}"] = xk
        g[f"y_{k}"] = yk
        g[f"z_{k}"] = zk
        g[f"u_{k}"] = uk
        uk_cols.append(f"u_{k}")

    g["Signal"] = g[uk_cols].mean(axis=1)
    return g


def add_signals(
    df: pd.DataFrame,
    price_col: str = "price_usd",
    t1: int = 12,
    t2: int = 168,
) -> pd.DataFrame:
    """
    Expects a panel with at least:
        datetime, fsym, price_usd
    """
    required = {"datetime", "fsym", price_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = (
        df.groupby("fsym", group_keys=False)
        .apply(lambda g: compute_signal_for_group(g, price_col=price_col, t1=t1, t2=t2))
        .reset_index(drop=True)
    )

    return out