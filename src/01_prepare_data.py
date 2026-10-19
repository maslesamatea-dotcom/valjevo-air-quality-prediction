"""Spaja dnevne PM10 vrijednosti sa SEPA stanica iz tri godisnja fajla u jednu tabelu (stanice prisutne u sve tri godine)."""
import re
import pandas as pd

RAW, OUT = "data/raw/", "data/processed/pm10_dnevno_2022_2024.csv"

def norm(c):
    c = re.sub(r"μg\.m-3|AOP|PM10|PM2\.5|korigovano|korig|ppb", "", str(c))
    return re.sub(r"\s+", " ", c).strip()

def clean(df):
    df.columns = ["date"] + [norm(c) for c in df.columns[1:]]
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

d23 = clean(pd.read_excel(RAW + "2023.xlsx", sheet_name="2023-pm10"))
d24 = clean(pd.read_excel(RAW + "2024.xlsx", sheet_name="2024-PM10"))

# 2022: list ima vise blokova stanica (drzavna mreza, gradske mreze...); koristimo samo prvi blok do 'Smederevo Radinac'
raw22 = pd.read_excel(RAW + "2022.xlsx", sheet_name="2022-pm10")
cols = list(raw22.columns)
end = [i for i, c in enumerate(cols) if "Smederevo Radinac" in str(c)][0]
use = [0] + [i for i in range(1, end + 1) if not str(cols[i]).startswith("Unnamed")]
d22 = raw22.iloc[:, use].copy()
d22.columns = ["date"] + [cols[i] for i in use[1:]]
d22 = clean(d22)

common = sorted(set(d22.columns[1:]) & set(d23.columns[1:]) & set(d24.columns[1:]))
out = pd.concat([d[["date"] + common] for d in (d22, d23, d24)], ignore_index=True).sort_values("date").reset_index(drop=True)
assert out["date"].is_unique and len(out) == 365 + 365 + 366
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"Sacuvano: {OUT} | {out.shape[0]} dana, {len(common)} stanica")
print("Stanice:", ", ".join(common))
