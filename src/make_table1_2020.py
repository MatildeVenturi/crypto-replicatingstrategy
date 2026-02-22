from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

import requests
import pandas as pd


@dataclass(frozen=True)
class Config:
    # IMPORTANT: usa la cartella corrente da cui lanci il comando (root repo)
    repo_root: Path = Path.cwd()
    data_processed_dir: Path = repo_root / "data" / "processed"
    tables_dir: Path = repo_root / "tables"

    base_url: str = "https://min-api.cryptocompare.com"
    exchange: str = "CCAGG"
    tsym: str = "USD"
    max_points_per_call: int = 2000

    # T = 4141 points inclusive => 4140 ore di differenza
    start_utc: datetime = datetime(2020, 2, 25, 0, 0, tzinfo=timezone.utc)
    end_utc: datetime = start_utc + timedelta(hours=4140)

    symbols: List[str] = ("BTC", "ETH", "DASH", "LTC", "MAID", "XMR", "XRP")


def _to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


def _auth_headers() -> Dict[str, str]:
    key = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()
    return {"authorization": f"Apikey {key}"} if key else {}


def fetch_histohour(fsym: str, cfg: Config, sleep_s: float = 0.2) -> pd.DataFrame:
    endpoint = f"{cfg.base_url}/data/v2/histohour"
    headers = _auth_headers()

    start_ts = _to_unix(cfg.start_utc)
    end_ts = _to_unix(cfg.end_utc)

    all_rows = []
    to_ts = end_ts

    while True:
        params = {
            "fsym": fsym,
            "tsym": cfg.tsym,
            
            "limit": cfg.max_points_per_call - 1,  # limit=N => N+1 punti
            "toTs": to_ts,
        }
        r = requests.get(endpoint, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        payload = r.json()

        if payload.get("Response") != "Success":
            raise RuntimeError(f"CryptoCompare error for {fsym}: {payload.get('Message')}")

        data = payload["Data"]["Data"]
        if not data:
            break

        all_rows.extend(data)
        oldest_ts = data[0]["time"]

        if oldest_ts <= start_ts:
            break

        to_ts = oldest_ts - 1
        time.sleep(sleep_s)

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("datetime")
    df = df.loc[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()

    df["fsym"] = fsym
    df.rename(columns={"close": "price_usd"}, inplace=True)

    expected = int((cfg.end_utc - cfg.start_utc).total_seconds() // 3600) + 1
    if len(df) != expected:
        raise ValueError(f"{fsym}: expected {expected} hours, got {len(df)}")

    return df[["datetime", "fsym", "price_usd", "open", "high", "low", "volumefrom", "volumeto"]]


def build_table1_summary(prices: pd.DataFrame) -> pd.DataFrame:
    g = prices.groupby("fsym")["price_usd"]
    table = pd.DataFrame({
        "N": g.size(),
        "mean": g.mean(),
        "std": g.std(ddof=1),
        "min": g.min(),
        "median": g.median(),
        "max": g.max(),
        "skew": g.apply(lambda s: s.skew()),
        "kurtosis": g.apply(lambda s: s.kurt()),  # excess kurtosis
    }).sort_index()
    return table


def main() -> None:
    cfg = Config()

    print("=== PATH CHECK ===")
    print("repo_root:", cfg.repo_root.resolve())
    print("data_processed_dir:", cfg.data_processed_dir.resolve())
    print("tables_dir:", cfg.tables_dir.resolve())

    cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sym in cfg.symbols:
        print(f"Fetching {sym} ...")
        frames.append(fetch_histohour(sym, cfg))

    prices = pd.concat(frames, ignore_index=True)

    out_data = cfg.data_processed_dir / "cryptocompare_hourly_usd_2020_T4141.csv"
    prices.to_csv(out_data, index=False)
    print("Saved processed data:", out_data.resolve())

    table1 = build_table1_summary(prices)

    out_csv = cfg.tables_dir / "table1_summary_stats_2020.csv"
    out_md = cfg.tables_dir / "table1_summary_stats_2020.md"
    table1.to_csv(out_csv)
    out_md.write_text(table1.to_markdown(), encoding="utf-8")

    print("Saved Table 1 (CSV):", out_csv.resolve())
    print("Saved Table 1 (MD):", out_md.resolve())


if __name__ == "__main__":
    main()