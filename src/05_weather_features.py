"""Valjevo: efekat meteoroloskih podataka (Open-Meteo / ERA5) na predikciju sutrasnjeg PM10."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

UP, OUT, LIM, SEED = "data/raw/", "results/", 50, 42
FULL = pd.date_range("2022-01-01", "2024-12-31", freq="D")

# ---------------- zagadjenje (isto kao ranije)
def series(file, sheet, key, positional_2022=False):
    d = pd.read_excel(UP + file, sheet_name=sheet)
    col = [c for c in d.columns if key in str(c)][0]
    v = pd.to_numeric(d[col], errors="coerce")
    idx = pd.date_range("2022-01-01", periods=len(d), freq="D") if positional_2022 else pd.to_datetime(d.iloc[:, 0], errors="coerce")
    return pd.Series(v.values, index=idx)
def combine(parts):
    s = pd.concat(parts); s = s[~s.index.isna()]; s = s[~s.index.duplicated(keep="first")]; return s.reindex(FULL)
pm10 = pd.read_csv("data/processed/pm10_dnevno_2022_2024.csv", parse_dates=["date"]).set_index("date")["Valjevo"].reindex(FULL)
pm25 = combine([series("2022.xlsx", "2022-pm2-5", "Valjevo"), series("2023.xlsx", "2023-PM2.5", "Valjevo"), series("2024.xlsx", "2024-PM2.5", "Valjevo")])
no2 = combine([series("2022.xlsx", "2022-no2", "Valjevo"), series("2023.xlsx", "2023-no2", "Valjevo"), series("2024.xlsx", "2024-NO2", "Valjevo")])
co = combine([series("2022.xlsx", "2022-co-max8h", "Valjevo", True), series("2023.xlsx", "2023-co_max8h", "Valjevo"), series("2024.xlsx", "2024-COmax8h", "Valjevo")])
raw = pd.DataFrame({"pm10": pm10, "pm25": pm25, "no2": no2, "co": co}).interpolate(limit=3, limit_area="inside")

# ---------------- meteorologija
w = pd.read_csv(UP + "open-meteo-valjevo-2022-2024.csv", skiprows=3, parse_dates=["time"]).set_index("time")
w.columns = [re.sub(r"\s*\(.*\)", "", c) for c in w.columns]
w = w.reindex(FULL)
print("meteo nedostaje:", int(w.isna().sum().sum()), "| korelacija sa PM10 istog dana:\n",
      pd.concat([w, raw.pm10], axis=1).corr()["pm10"].drop("pm10").round(2).to_string())
w["wind_sin"] = np.sin(np.deg2rad(w.wind_direction_10m_dominant)); w["wind_cos"] = np.cos(np.deg2rad(w.wind_direction_10m_dominant))
w["temp_range"] = w.temperature_2m_max - w.temperature_2m_min
w["precip_3d"] = w.precipitation_sum.rolling(3, min_periods=1).sum()
w["pressure_change"] = w.surface_pressure_mean - w.surface_pressure_mean.shift(1)
WCOLS = ["temperature_2m_mean", "temperature_2m_min", "precipitation_sum", "precip_3d", "wind_speed_10m_max", "wind_gusts_10m_max",
         "wind_sin", "wind_cos", "surface_pressure_mean", "pressure_change", "relative_humidity_2m_mean", "cloud_cover_mean", "temp_range"]

# ---------------- feature-i
X = pd.DataFrame(index=raw.index)
for k in (0, 1, 2, 3, 6): X[f"pm10_lag{k}"] = raw.pm10.shift(k)
X["pm10_mean3"] = raw.pm10.rolling(3, min_periods=2).mean(); X["pm10_mean7"] = raw.pm10.rolling(7, min_periods=4).mean()
X["pm10_max7"] = raw.pm10.rolling(7, min_periods=4).max(); X["pm10_diff1"] = raw.pm10 - raw.pm10.shift(1)
X["pm25_lag0"], X["no2_lag0"], X["co_lag0"] = raw.pm25, raw.no2, raw.co
td = X.index + pd.Timedelta(days=1)
X["mjesec_sin"], X["mjesec_cos"] = np.sin(2*np.pi*td.month/12), np.cos(2*np.pi*td.month/12)
X["dan_sedmice"], X["vikend"] = td.dayofweek, (td.dayofweek >= 5).astype(int)
X["grijna_sezona"] = td.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
POLL = list(X.columns)
for c in WCOLS: X[f"w_danas_{c}"] = w[c]                  # meteo na dan t (poznato)
for c in WCOLS: X[f"w_sutra_{c}"] = w[c].shift(-1)        # meteo na dan t+1 (= 'savrsena prognoza')
SETS = {"A) samo zagađenje": POLL,
        "B) + meteo (danas)": POLL + [f"w_danas_{c}" for c in WCOLS],
        "C) + meteo (sutra, 'prognoza')": POLL + [f"w_danas_{c}" for c in WCOLS] + [f"w_sutra_{c}" for c in WCOLS]}
y = raw.pm10.shift(-1)
D = X.assign(y=y, target_date=td).dropna(subset=["y", "pm10_lag0"])
tr, te = D[D.target_date < "2024-01-01"], D[D.target_date >= "2024-01-01"]
ytr, yte = tr.y, te.y; ctr, cte = (ytr > LIM).astype(int), (yte > LIM).astype(int)
win = te.target_date.dt.month.isin([10, 11, 12, 1, 2, 3]).values
base = te.pm10_lag0.values
print(f"\nbaseline MAE: cijela {mean_absolute_error(yte, base):.2f} | grijna {mean_absolute_error(yte[win], base[win]):.2f}")

mk_reg = {"Ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
          "Random Forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, random_state=SEED, n_jobs=-1)),
          "HistGB": lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=.05, max_iter=300, l2_regularization=1., random_state=SEED)}
mk_clf = {"LogReg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", C=.3, max_iter=3000)),
          "Random Forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(400, min_samples_leaf=3, max_features=0.5, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
          "HistGB": lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=.05, max_iter=300, l2_regularization=1., class_weight="balanced", random_state=SEED)}
rows, bestp = [], {}
for sn, cols in SETS.items():
    for mn in mk_reg:
        m = mk_reg[mn]().fit(tr[cols], ytr); p = m.predict(te[cols]); bestp[(sn, mn)] = p
        r = dict(skup=sn, model=mn, MAE=mean_absolute_error(yte, p), RMSE=mean_squared_error(yte, p) ** .5, R2=r2_score(yte, p),
                 MAE_grijna=mean_absolute_error(yte[win], p[win]), R2_grijna=r2_score(yte[win], p[win]))
        cn = "LogReg" if mn == "Ridge" else mn
        c = mk_clf[cn]().fit(tr[cols], ctr); pr = c.predict_proba(te[cols])[:, 1]; yp = (pr >= .5).astype(int)
        r.update(F1=f1_score(cte, yp), precision=precision_score(cte, yp), recall=recall_score(cte, yp), AUC=roc_auc_score(cte, pr))
        rows.append(r)
R = pd.DataFrame(rows).round(3); R.to_csv(OUT + "rezultati_meteo.csv", index=False)
pd.set_option("display.width", 220); print("\n", R.to_string(index=False))

# unakrsna provjera (vremenska) za RF na skupovima A i C
tscv = TimeSeriesSplit(n_splits=4)
for sn in (list(SETS)[0], list(SETS)[2]):
    cv = [mean_absolute_error(ytr.iloc[b], mk_reg["Random Forest"]().fit(tr[SETS[sn]].iloc[a], ytr.iloc[a]).predict(tr[SETS[sn]].iloc[b])) for a, b in tscv.split(tr)]
    print(f"CV MAE RF {sn}: {np.mean(cv):.2f} ± {np.std(cv):.2f}")
cvb = [mean_absolute_error(ytr.iloc[b], tr.pm10_lag0.iloc[b]) for a, b in tscv.split(tr)]
print(f"CV MAE baseline: {np.mean(cvb):.2f} ± {np.std(cvb):.2f}")

# graf: MAE po skupovima
rf = R[R.model == "Random Forest"].set_index("skup")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, col, ttl, bl in ((ax[0], "MAE", "MAE, cijela 2024", mean_absolute_error(yte, base)),
                        (ax[1], "MAE_grijna", "MAE, grijna sezona", mean_absolute_error(yte[win], base[win]))):
    rf[col].plot.bar(ax=a, color="#1f3864", rot=12); a.axhline(bl, color="crimson", ls="--", label="Baseline (sutra = danas)")
    a.set_title(ttl + " (Random Forest)"); a.set_xlabel(""); a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + "8_meteo_efekat.png", dpi=130); plt.close()

# znacaj feature-a za RF, skup B
cols = SETS["B) + meteo (danas)"]; m = mk_reg["Random Forest"]().fit(tr[cols], ytr)
imp = pd.Series(m[-1].feature_importances_, index=cols).sort_values()[-12:]
fig, ax = plt.subplots(figsize=(7.5, 4.6)); imp.plot.barh(ax=ax, color="#1f3864"); ax.set_title("Značaj feature-a, RF, skup B (zagađenje + meteo danas)")
fig.tight_layout(); fig.savefig(OUT + "9_znacaj_meteo.png", dpi=130)
print("\nTop (B):", imp.sort_values(ascending=False).round(3).head(8).to_dict())
