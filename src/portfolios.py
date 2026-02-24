def compute_ts_portfolio(df_with_signal: pd.DataFrame) -> pd.DataFrame:
    df = df_with_signal.sort_values(["datetime", "fsym"]).copy()
    N = df["fsym"].nunique()

    df["r"] = df.groupby("fsym")["P"].pct_change()  # (P_t/P_{t-1}-1)
    df["pos"] = df.groupby("fsym")["Signal"].shift(1) / N

    df["ts_contrib"] = df["pos"] * df["r"]
    ts = df.groupby("datetime")["ts_contrib"].sum().to_frame("R_TS")

    ts["R_TS_cum"] = (1 + ts["R_TS"]).cumprod()
    return ts

def compute_cs_portfolio(df_with_signal: pd.DataFrame) -> pd.DataFrame:
    df = df_with_signal.sort_values(["datetime", "fsym"]).copy()
    df["r"] = df.groupby("fsym")["P"].pct_change()

    # weights decided at t-1, applied to return at t
    def weights_at_time(g):
        g = g.copy()
        g["w"] = 0.0
        s = g["Signal"]
        top = s.nlargest(3).index
        bot = s.nsmallest(3).index
        g.loc[top, "w"] =  1.0/3.0
        g.loc[bot, "w"] = -1.0/3.0
        return g

    df = df.groupby("datetime", group_keys=False).apply(weights_at_time)
    df["w_lag"] = df.groupby("fsym")["w"].shift(1)  # use info at t-1
    df["cs_contrib"] = df["w_lag"] * df["r"]

    cs = df.groupby("datetime")["cs_contrib"].sum().to_frame("R_CS")
    cs["R_CS_cum"] = (1 + cs["R_CS"]).cumprod()
    return cs