"""Provjera na vise stanica: ista PM10 'samo-prosla-mjerenja' postavka za Valjevo, Beograd Stari grad, Nis i Beocin."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.metrics import (mean_absolute_error, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

OUT, LIM, SEED = "results/", 50, 42
STATIONS = ["Valjevo", "Beograd Stari grad", "Niš O.š. Sveti Sava", "Beočin Centar"]
df = pd.read_csv("data/processed/pm10_dnevno_2022_2024.csv", parse_dates=["date"]).set_index("date")

def build(s):
    s = s.interpolate(limit=3, limit_area="inside")
    X = pd.DataFrame(index=s.index)
    for k in (0, 1, 2, 3, 6): X[f"lag{k}"] = s.shift(k)
    X["mean3"] = s.rolling(3, min_periods=2).mean()
    X["mean7"] = s.rolling(7, min_periods=4).mean()
    X["max7"] = s.rolling(7, min_periods=4).max()
    X["diff1"] = s - s.shift(1)
    td = X.index + pd.Timedelta(days=1)
    X["m_sin"], X["m_cos"] = np.sin(2*np.pi*td.month/12), np.cos(2*np.pi*td.month/12)
    X["dow"], X["vikend"] = td.dayofweek, (td.dayofweek >= 5).astype(int)
    X["grijna"] = td.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    y = s.shift(-1)
    D = X.assign(y=y, target_date=td).dropna(subset=["y", "lag0"])
    return D

rows = []
for st in STATIONS:
    D = build(df[st]); tr, te = D[D.target_date < "2024-01-01"], D[D.target_date >= "2024-01-01"]
    feats = [c for c in D.columns if c not in ("y", "target_date")]
    Xtr, Xte, ytr, yte = tr[feats], te[feats], tr.y, te.y
    ctr, cte = (ytr > LIM).astype(int), (yte > LIM).astype(int)
    pos_tr, pos_te = ctr.mean(), cte.mean()
    imp = lambda m: make_pipeline(SimpleImputer(strategy="median"), m)
    regs = {"Baseline": None,
            "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
            "RandomForest": imp(RandomForestRegressor(400, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
            "HistGB": HistGradientBoostingRegressor(max_depth=4, learning_rate=.05, max_iter=300, l2_regularization=1., random_state=SEED)}
    clfs = {"Baseline": None,
            "LogReg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
            "RandomForest": imp(RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
            "HistGB": HistGradientBoostingClassifier(max_depth=4, learning_rate=.05, max_iter=300, l2_regularization=1., class_weight="balanced", random_state=SEED)}
    base_mae = mean_absolute_error(yte, Xte.lag0)
    pairs = [("Baseline", "Baseline"), ("Ridge", "LogReg"), ("RandomForest", "RandomForest"), ("HistGB", "HistGB")]
    for rn, cn in pairs:
        p = Xte.lag0.values if regs[rn] is None else regs[rn].fit(Xtr, ytr).predict(Xte)
        mae = mean_absolute_error(yte, p)
        n = "Baseline" if rn == "Baseline" else ("Linearni (Ridge/LogReg)" if rn == "Ridge" else rn)
        if clfs[cn] is None:
            pr = Xte.lag0.values; yp = (pr > LIM).astype(int)
        else:
            clfs[cn].fit(Xtr, ctr); pr = clfs[cn].predict_proba(Xte)[:, 1]; yp = (pr >= .5).astype(int)
        rows.append(dict(stanica=st, model=n, udio_prekoracenja_test=round(pos_te, 3), MAE=round(mae, 2),
                         MAE_vs_baseline_pct=round(100*(base_mae-mae)/base_mae, 1),
                         precision=round(precision_score(cte, yp, zero_division=0), 3),
                         recall=round(recall_score(cte, yp, zero_division=0), 3),
                         F1=round(f1_score(cte, yp, zero_division=0), 3),
                         AUC=round(roc_auc_score(cte, pr), 3), PR_AUC=round(average_precision_score(cte, pr), 3)))
R = pd.DataFrame(rows); R.to_csv(OUT + "rezultati_vise_stanica.csv", index=False)
pd.set_option("display.width", 200)
print(R.to_string(index=False))

# graf: poboljsanje MAE u odnosu na baseline i F1 po stanicama
piv_mae = R[R.model != "Baseline"].pivot(index="stanica", columns="model", values="MAE_vs_baseline_pct").loc[STATIONS]
piv_f1 = R.pivot(index="stanica", columns="model", values="F1").loc[STATIONS]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
piv_mae.plot.bar(ax=ax[0], rot=15); ax[0].axhline(0, color="k", lw=.8)
ax[0].set_title("Smanjenje MAE u odnosu na baseline (%)"); ax[0].set_xlabel(""); ax[0].grid(alpha=.3)
piv_f1[["Baseline", "Linearni (Ridge/LogReg)", "RandomForest", "HistGB"]].plot.bar(ax=ax[1], rot=15)
ax[1].set_title("F1 za klasu 'sutra > 50 µg/m³'"); ax[1].set_xlabel(""); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + "7_vise_stanica.png", dpi=130)
