"""PM10 Valjevo: predikcija sutrasnjeg dnevnog PM10 (regresija) i prekoracenja limita 50 ug/m3 (klasifikacija)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

UP, OUT, LIM, SEED = "data/raw/", "results/", 50, 42
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    USE_CB = True
except Exception:
    USE_CB = False
GB_NAME = "CatBoost" if USE_CB else "HistGradientBoosting"

# ---------------------------------------------------------------- podaci
FULL = pd.date_range("2022-01-01", "2024-12-31", freq="D")
def series(file, sheet, key, positional_2022=False):
    d = pd.read_excel(UP + file, sheet_name=sheet)
    col = [c for c in d.columns if key in str(c)][0]
    v = pd.to_numeric(d[col], errors="coerce")
    if positional_2022:                      # 2022 CO list ima pomijesane dan/mjesec datume -> pozicija
        idx = pd.date_range("2022-01-01", periods=len(d), freq="D")
    else:
        idx = pd.to_datetime(d.iloc[:, 0], errors="coerce")
    return pd.Series(v.values, index=idx)

def combine(parts):
    s = pd.concat(parts)
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(FULL)

pm10 = pd.read_csv("data/processed/pm10_dnevno_2022_2024.csv", parse_dates=["date"]).set_index("date")["Valjevo"].reindex(FULL)
pm25 = combine([series("2022.xlsx", "2022-pm2-5", "Valjevo"), series("2023.xlsx", "2023-PM2.5", "Valjevo"), series("2024.xlsx", "2024-PM2.5", "Valjevo")])
no2  = combine([series("2022.xlsx", "2022-no2", "Valjevo"), series("2023.xlsx", "2023-no2", "Valjevo"), series("2024.xlsx", "2024-NO2", "Valjevo")])
so2  = combine([series("2022.xlsx", "2022-so2", "Valjevo"), series("2023.xlsx", "2023-so2", "Valjevo"), series("2024.xlsx", "2024-SO2", "Valjevo")])
co   = combine([series("2022.xlsx", "2022-co-max8h", "Valjevo", True), series("2023.xlsx", "2023-co_max8h", "Valjevo"), series("2024.xlsx", "2024-COmax8h", "Valjevo")])
raw = pd.DataFrame({"pm10": pm10, "pm25": pm25, "no2": no2, "so2": so2, "co": co})
print("nedostaje %:\n", (100 * raw.isna().mean()).round(1).to_string())
print("korelacija sa PM10:\n", raw.corr()["pm10"].round(2).to_string())

# ---------------------------------------------------------------- feature-i (samo prosla/dana: t i ranije)
raw = raw.interpolate(limit=3, limit_area="inside")
X = pd.DataFrame(index=raw.index)
for k in (0, 1, 2, 3, 6):
    X[f"pm10_lag{k}"] = raw["pm10"].shift(k)
X["pm10_mean3"] = raw["pm10"].rolling(3, min_periods=2).mean()
X["pm10_mean7"] = raw["pm10"].rolling(7, min_periods=4).mean()
X["pm10_max7"]  = raw["pm10"].rolling(7, min_periods=4).max()
X["pm10_diff1"] = raw["pm10"] - raw["pm10"].shift(1)
X["pm25_lag0"], X["pm25_lag1"] = raw["pm25"], raw["pm25"].shift(1)
X["no2_lag0"], X["so2_lag0"], X["co_lag0"] = raw["no2"], raw["so2"], raw["co"]
tgt_date = X.index + pd.Timedelta(days=1)          # kalendar CILJNOG (sutrasnjeg) dana
X["mjesec_sin"] = np.sin(2 * np.pi * tgt_date.month / 12)
X["mjesec_cos"] = np.cos(2 * np.pi * tgt_date.month / 12)
X["dan_sedmice"] = tgt_date.dayofweek
X["vikend"] = (tgt_date.dayofweek >= 5).astype(int)
X["grijna_sezona"] = tgt_date.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
y = raw["pm10"].shift(-1)                          # PM10 sutra
yc = (y > LIM).astype(float).where(y.notna())
D = X.assign(y=y, yc=yc, target_date=tgt_date).dropna(subset=["y", "pm10_lag0"])
train = D[D.target_date < "2024-01-01"]; test = D[D.target_date >= "2024-01-01"]
feats = [c for c in X.columns]
print(f"\ntrening: {len(train)} dana (cilj 2022-2023) | test: {len(test)} dana (cilj 2024)")
print("udio dana >50  trening:", round(train.yc.mean(), 3), "| test:", round(test.yc.mean(), 3))
Xtr, Xte, ytr, yte = train[feats], test[feats], train.y, test.y
winter = test.target_date.dt.month.isin([10, 11, 12, 1, 2, 3]).values

# ---------------------------------------------------------------- REGRESIJA
def mets(yt, yp):
    return dict(MAE=mean_absolute_error(yt, yp), RMSE=mean_squared_error(yt, yp) ** .5, R2=r2_score(yt, yp))
def gb_reg():
    return CatBoostRegressor(iterations=400, depth=4, learning_rate=.05, verbose=0, random_seed=SEED) if USE_CB else \
           HistGradientBoostingRegressor(max_depth=4, learning_rate=.05, max_iter=300, l2_regularization=1., random_state=SEED)
regs = {
    "Baseline (sutra = danas)": None,
    "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(400, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
    GB_NAME: gb_reg(),
}
rows, preds = [], {}
for n, m in regs.items():
    p = Xte["pm10_lag0"].values if m is None else m.fit(Xtr, ytr).predict(Xte)
    preds[n] = p
    r = {"model": n, **{k + " (cijela 2024)": v for k, v in mets(yte, p).items()},
         **{k + " (grijna sezona)": v for k, v in mets(yte[winter], p[winter]).items()}}
    rows.append(r)
R = pd.DataFrame(rows).set_index("model").round(2)
print("\n=== REGRESIJA (test 2024) ===\n", R.to_string())
R.to_csv(OUT + "rezultati_regresija.csv")

# vremenska unakrsna validacija na treningu (provjera stabilnosti)
tscv = TimeSeriesSplit(n_splits=4)
cv = {}
for n, m in regs.items():
    if m is None: continue
    maes = []
    for a, b in tscv.split(Xtr):
        maes.append(mean_absolute_error(ytr.iloc[b], m.fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b])))
    cv[n] = (np.mean(maes), np.std(maes))
base_cv = []
for a, b in tscv.split(Xtr):
    base_cv.append(mean_absolute_error(ytr.iloc[b], Xtr.iloc[b]["pm10_lag0"]))
print("\nMAE CV (TimeSeriesSplit, trening):  baseline %.1f ± %.1f" % (np.mean(base_cv), np.std(base_cv)),
      {k: "%.1f ± %.1f" % v for k, v in cv.items()})

# ---------------------------------------------------------------- KLASIFIKACIJA
yctr, ycte = train.yc.astype(int), test.yc.astype(int)
def gb_clf():
    return CatBoostClassifier(iterations=400, depth=4, learning_rate=.05, verbose=0, random_seed=SEED, auto_class_weights="Balanced") if USE_CB else \
           HistGradientBoostingClassifier(max_depth=4, learning_rate=.05, max_iter=300, l2_regularization=1., class_weight="balanced", random_state=SEED)
clfs = {
    "Baseline (sutra > 50 ako je danas > 50)": None,
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
    GB_NAME: gb_clf(),
}
rows, probs = [], {}
def cm(yt, yp, pr=None):
    d = dict(Accuracy=accuracy_score(yt, yp), Precision=precision_score(yt, yp, zero_division=0),
             Recall=recall_score(yt, yp, zero_division=0), F1=f1_score(yt, yp, zero_division=0))
    d["AUC"] = roc_auc_score(yt, pr) if pr is not None else np.nan
    return d
for n, m in clfs.items():
    if m is None:
        yp = (Xte["pm10_lag0"].values > LIM).astype(int); pr = Xte["pm10_lag0"].values
    else:
        m.fit(Xtr, yctr); pr = m.predict_proba(Xte)[:, 1]; yp = (pr >= .5).astype(int)
    probs[n] = (yp, pr)
    rows.append({"model": n, **cm(ycte, yp, pr)})
C = pd.DataFrame(rows).set_index("model").round(3)
print("\n=== KLASIFIKACIJA: sutra > 50 ug/m3 (test 2024) ===\n", C.to_string())
C.to_csv(OUT + "rezultati_klasifikacija.csv")
print("\nkonfuzija (%s):\n" % GB_NAME, confusion_matrix(ycte, probs[GB_NAME][0]))

# ---------------------------------------------------------------- grafici
best = "Random Forest" if R.loc["Random Forest", "MAE (cijela 2024)"] <= R.loc[GB_NAME, "MAE (cijela 2024)"] else GB_NAME
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test.target_date, yte, color="#1f3864", lw=1, label="Stvarno")
ax.plot(test.target_date, preds[best], color="#e07b00", lw=1, label=f"Predikcija ({best})")
ax.plot(test.target_date, preds["Baseline (sutra = danas)"], color="gray", lw=.7, alpha=.6, label="Baseline (sutra = danas)")
ax.axhline(LIM, color="crimson", ls="--", lw=1)
ax.set_title("PM10 Valjevo, test 2024: stvarno vs predikcija"); ax.set_ylabel("µg/m³"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + "5_predikcija_2024.png", dpi=130); plt.close()

rf = regs["Random Forest"]
imp = pd.Series(rf[-1].feature_importances_, index=feats).sort_values()[-12:]
fig, ax = plt.subplots(figsize=(7, 4.5)); imp.plot.barh(ax=ax, color="#1f3864")
ax.set_title("Značaj feature-a (Random Forest)"); fig.tight_layout(); fig.savefig(OUT + "6_znacaj_featurea.png", dpi=130); plt.close()
print("\nTop feature-i:", imp.sort_values(ascending=False).round(3).head(8).to_dict())
print("koristen gradient boosting:", GB_NAME)
