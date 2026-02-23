import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/cryptocompare_hourly_usd_2020_T4141.csv")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

# Keep only BTC
df = df[df["fsym"] == "BTC"].copy()

df = df.dropna(subset=["datetime", "price_usd"])
df = df.sort_values("datetime").reset_index(drop=True)

print(df["price_usd"].describe())

r = df["price_usd"].pct_change()

print("Max hourly return:", r.max())
print("Min hourly return:", r.min())
print(">|50%| hourly moves:", (r.abs() > 0.5).sum())
print(">|100%| hourly moves:", (r.abs() > 1.0).sum())