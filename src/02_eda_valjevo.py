import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/pm10_dnevno_2022_2024.csv", parse_dates=["date"])
s = df.set_index("date")["Valjevo"]
OUT = "results/"
LIM = 50

# --- osnovne brojke
print("dana:", len(s), "| nedostaje:", s.isna().sum(), f"({100*s.isna().mean():.1f}%)")
print(s.describe().round(1).to_string())
print("dana > 50:", int((s > LIM).sum()), f"({100*(s>LIM).sum()/s.notna().sum():.1f}%)")
# najduže rupe
isn = s.isna().astype(int)
runs = isn.groupby((isn.diff() != 0).cumsum()).agg(['sum', lambda x: x.index[0]])
runs.columns = ['len', 'start']
print("najduže rupe:\n", runs[isn.groupby((isn.diff() != 0).cumsum()).first() == 1].sort_values('len', ascending=False).head(5))

# --- 1. vremenska serija
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(s.index, s.values, lw=0.8, color="#1f3864")
ax.axhline(LIM, color="crimson", ls="--", lw=1, label="Dnevni limit 50 µg/m³")
ax.set_title("PM10 u Valjevu, dnevni prosjek 2022–2024")
ax.set_ylabel("µg/m³"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + "1_vremenska_serija.png", dpi=130); plt.close()

# --- 2. sezonalnost po mjesecima
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
m = [s[s.index.month == k].dropna().values for k in range(1, 13)]
axs[0].boxplot(m, tick_labels=range(1, 13), showfliers=False)
axs[0].axhline(LIM, color="crimson", ls="--", lw=1)
axs[0].set_title("PM10 po mjesecima"); axs[0].set_xlabel("mjesec"); axs[0].set_ylabel("µg/m³")
share = [100 * (x > LIM).mean() for x in m]
axs[1].bar(range(1, 13), share, color="#1f3864")
axs[1].set_title("% dana iznad limita po mjesecima"); axs[1].set_xlabel("mjesec"); axs[1].set_ylabel("%")
for a in axs: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + "2_sezonalnost.png", dpi=130); plt.close()
print("\n% dana > 50 po mjesecima:", [round(x) for x in share])

# --- 3. raspodjela + dan u sedmici
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(s.dropna(), bins=40, color="#1f3864")
axs[0].axvline(LIM, color="crimson", ls="--"); axs[0].set_title("Raspodjela PM10"); axs[0].set_xlabel("µg/m³")
dow = s.groupby(s.index.dayofweek).mean()
axs[1].bar(["Pon","Uto","Sri","Čet","Pet","Sub","Ned"], dow.values, color="#1f3864")
axs[1].set_title("Prosječan PM10 po danu u sedmici"); axs[1].set_ylabel("µg/m³")
for a in axs: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + "3_raspodjela_dani.png", dpi=130); plt.close()
print("po danu u sedmici:", dow.round(1).to_dict())
print("skewness:", round(s.skew(), 2))

# --- 4. autokorelacija + danas vs sutra
ac = [s.autocorr(k) for k in range(1, 15)]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].bar(range(1, 15), ac, color="#1f3864"); axs[0].set_title("Autokorelacija (koliko prošlost objašnjava danas)")
axs[0].set_xlabel("pomak u danima"); axs[0].grid(alpha=.3)
axs[1].scatter(s.values[:-1], s.values[1:], s=8, alpha=.5, color="#1f3864")
axs[1].plot([0, 300], [0, 300], color="crimson", lw=1)
axs[1].set_xlabel("PM10 danas"); axs[1].set_ylabel("PM10 sutra"); axs[1].set_title("Danas vs. sutra"); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT + "4_autokorelacija.png", dpi=130); plt.close()
print("autokorelacija lag1..3,7:", [round(ac[i], 2) for i in (0, 1, 2, 6)])

# --- klase: sutra > 50
y = (s.shift(-1) > LIM).astype(float).where(s.shift(-1).notna())
print("klasa 'sutra>50':", y.value_counts(normalize=True).round(3).to_dict())
for yr in (2022, 2023, 2024):
    q = s[s.index.year == yr]
    print(yr, "prosjek", round(q.mean(), 1), "| dana>50:", int((q > LIM).sum()))
