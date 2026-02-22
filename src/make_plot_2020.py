from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    repo_root = Path.cwd()

    data_path = repo_root / "data" / "processed" / "cryptocompare_hourly_usd_2020_T4141.csv"
    out_dir = repo_root / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["datetime"])

    # Se vuoi l'asse x = "Hours" (0..4140) come nel paper:
    df = df.sort_values(["fsym", "datetime"])
    df["hour_index"] = df.groupby("fsym").cumcount()

    # Pivot: righe=ore, colonne=crypto, valori=price
    wide = df.pivot(index="hour_index", columns="fsym", values="price_usd").sort_index()

    # Indexed price: divide ogni colonna per il primo valore
    indexed = wide / wide.iloc[0]

    # Plot
    plt.figure()
    indexed.plot(ax=plt.gca(), legend=True)

    plt.xlabel("Hours")
    plt.ylabel("Indexed Price")

    # Legenda come nel paper (solo nomi)
    plt.legend(title=None)

    out_png = out_dir / "figure1_indexed_prices_2020.png"
    out_pdf = out_dir / "figure1_indexed_prices_2020.pdf"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print("Saved:", out_png.resolve())
    print("Saved:", out_pdf.resolve())


if __name__ == "__main__":
    main()