import numpy as np
import pandas as pd

SHORTS = (8, 16, 32)
LONGS  = (24, 48, 96)   # paired by index with SHORTS

def compute_indexed_price(g: pd.DataFrame, price_col="price_usd") -> pd.Series:
    p0 = g[price_col].iloc[0]
    return g[price_col] / p0

def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=numer.index)
    mask = denom.ne(0) & numer.ne(0)
    out.loc[mask] = numer.loc[mask] / denom.loc[mask]
    return out

def compute_signal_for_group(g: pd.DataFrame,
                             price_col="price_usd",
                             t1=12, t2=168) -> pd.DataFrame:
    """
    Returns g with columns:
      P (indexed), Signal, and optionally intermediate columns if you keep them.
    """
    g = g.sort_values("datetime").copy()
    g["P"] = compute_indexed_price(g, price_col=price_col)

    # rolling std of indexed price for step 1
    sigma_P = g["P"].rolling(t1, min_periods=t1).std()

    u_list = []

    for nk_s, nk_l in zip(SHORTS, LONGS):
        ema_s = g["P"].ewm(span=nk_s, adjust=False).mean()
        ema_l = g["P"].ewm(span=nk_l, adjust=False).mean()

        # Eq (2): base signal
        xk = (ema_s - ema_l) / g["P"]

        # Eq (3): first normalization
        yk = safe_div(xk, sigma_P)

        # Eq (4): second normalization
        sigma_y = yk.rolling(t2, min_periods=t2).std()
        zk = safe_div(yk, sigma_y)

        # Eq (5): scaling (stable form)
        uk = 2.0 * np.tanh(zk / 8.0)
        u_list.append(uk)

    g["Signal"] = (u_list[0] + u_list[1] + u_list[2]) / 3.0
    return g

def add_signals(df: pd.DataFrame,
                price_col="price_usd",
                t1=12, t2=168) -> pd.DataFrame:
    return (df.groupby("fsym", group_keys=False)
              .apply(lambda g: compute_signal_for_group(g, price_col, t1, t2)))