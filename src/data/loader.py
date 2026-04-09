# src/data/loader.py

import pandas as pd
from pathlib import Path

MISC_PATTERN = r"^(POST, DOT, AMAZON, BANK\s*CHARGES, CRUK, gift_, POSTAGE, M$, D$)"


def _resolve_path(file_path: str) -> Path:
    """Search locations for the Excel file."""
    for candidate in [
        Path(file_path),
        Path("dataset") / file_path,
        Path("dataset/raw") / file_path,
        Path("..") / file_path,
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find '{file_path}'. Searched: dataset/, dataset/raw/, ../, cwd."
    )


def load_and_clean_uci(
    scenario: str = "gross",
    file_path: str = "Online Retail.xlsx",
) -> pd.DataFrame:
    """
    Load and clean the UCI Online Retail dataset.
    scenario  : "gross"  -> Scenario 1 (max risk, positive quantities only)
                "netted" -> Scenario 2 (refunds/ cancellations + partial cancellations)
    """
    path = _resolve_path(file_path)
    df_raw = pd.read_excel(
        path,
        engine="openpyxl",
        parse_dates=["InvoiceDate"],
        dtype={
            "StockCode" : str,
            "InvoiceNo" : str,
            "CustomerID": str,
        },
    )

    df_raw["Quantity"]  = pd.to_numeric(df_raw["Quantity"],  errors="coerce")
    df_raw["UnitPrice"] = pd.to_numeric(df_raw["UnitPrice"], errors="coerce")

    # SCENARIO 1
    if scenario == "gross":
        df = (
            df_raw
                .dropna(subset=["CustomerID"])
                .query("Quantity > 0 and UnitPrice > 0")
                .astype({"StockCode": str})
                .assign(
                    OrderValue=lambda x: x.Quantity * x.UnitPrice,      # was OrderValue_GBP
                    OrderValue_GBP=lambda x: x.Quantity * x.UnitPrice,  # keep alias too
                    SKU=lambda x: x.StockCode.str.strip(),
                    Date=lambda x: x.InvoiceDate.dt.date
                )
                .reset_index(drop=True)
            )
        print(f"SCENARIO 1 (GROSS): {len(df):,} rows loaded")
        return df

    # SCENARIO 2
    df_raw["Date"] = df_raw["InvoiceDate"].dt.date
    df_raw["SKU"]  = df_raw["StockCode"].str.strip()

    real_mask = (
        ~df_raw["StockCode"].str.upper().str.match(MISC_PATTERN, na=False)
        & df_raw["StockCode"].str.strip().ne("")
        & df_raw["UnitPrice"].gt(0)
        & df_raw["CustomerID"].notna()
    )
    real_df = df_raw[real_mask].copy()
    gross_picks   = real_df[real_df["Quantity"] > 0].copy()
    cancellations = real_df[real_df["Quantity"] < 0].copy()

    for frame in (gross_picks, cancellations):
        frame["match_key"] = frame["CustomerID"] + "-" + frame["SKU"]

    # Net quantity 
    all_rows = pd.concat(
        [gross_picks[["match_key", "Quantity"]],
         cancellations[["match_key", "Quantity"]]],
        ignore_index=True,
    )
    net_qty_per_key = all_rows.groupby("match_key", sort=False)["Quantity"].sum()
    positive_keys   = net_qty_per_key[net_qty_per_key > 0]

    filtered_gross = gross_picks[gross_picks["match_key"].isin(positive_keys.index)]
    max_idx        = filtered_gross.groupby("match_key", sort=False)["Quantity"].idxmax()
    net_sales      = filtered_gross.loc[max_idx].copy()

    # Attach net quantities and values
    net_sales["NetQuantity"]   = net_sales["match_key"].map(positive_keys)
    net_sales["NetOrderValue"] = net_sales["NetQuantity"] * net_sales["UnitPrice"]

    df = (
        net_sales[[
            "NetQuantity", "NetOrderValue",
            "SKU", "Date", "StockCode",
            "InvoiceDate", "UnitPrice",
        ]]
        .rename(columns={
            "NetQuantity"   : "Quantity",
            "NetOrderValue" : "OrderValue_GBP",
        })
        .query("Quantity > 0")
        .reset_index(drop=True)
    )
    df["OrderValue"] = df["OrderValue_GBP"]  # keep alias for consistency with scenario 1   ]

    print(
        f"SCENARIO 2 (NETTED): {len(df):,} rows after netting "
        f"({len(gross_picks):,} gross picks, {len(cancellations):,} cancellations removed)"
    )
    return df