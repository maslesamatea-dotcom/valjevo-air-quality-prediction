# Predviđanje PM10 zagađenja u Valjevu (otvoreni podaci Srbije)

Projekat predviđa **sutrašnju dnevnu koncentraciju PM10** (sitne čestice) i **da li će biti prekoračen dnevni limit od 50 µg/m³** u Valjevu. Koristi otvorene podatke o kvalitetu vazduha Agencije za zaštitu životne sredine (SEPA, portal data.gov.rs) za 2022–2024. i meteorološke podatke (Open-Meteo / ERA5).

Autor: Matea Masleša

## Zašto Valjevo

Od 16 stanica sa kompletnim nizom kroz sve tri godine Valjevo ima jasnu sezonalnost (zimski prosjek ≈ 82 µg/m³, ljetni ≈ 27 µg/m³) i veliki broj dana iznad limita (411 od 1.077 mjerenja, 38%), što daje dovoljno primjera za oba zadatka.

## Podaci

| Izvor | Sadržaj | Fajl |
|---|---|---|
| SEPA / data.gov.rs | Dnevne vrijednosti PM10, PM2.5, NO₂, SO₂, CO po stanicama, 2022–2024. | `data/raw/2022.xlsx`, `2023.xlsx`, `2024.xlsx` |
| Open-Meteo (ERA5 reanaliza) | Dnevna temperatura, padavine, vjetar, pritisak, vlažnost, oblačnost | `data/raw/open-meteo-valjevo-2022-2024.csv` |

`src/01_prepare_data.py` spaja PM10 u jednu tabelu (`data/processed/`) sa 16 stanica prisutnih u sve tri godine.
Provjerite licencu SEPA skupova na portalu data.gov.rs prije ponovne upotrebe. Meteorološki podaci: Open-Meteo.com, licenca CC BY 4.0 (izvor: Copernicus ERA5).

## Metodologija

- **Cilj:** PM10 sutra (regresija) i PM10 sutra > 50 µg/m³ (klasifikacija).
- **Feature-i** (koriste samo informacije do današnjeg dana): PM10 od prethodnih dana (lag), klizni prosjeci i maksimum, PM2.5, NO₂, CO, kalendar sutrašnjeg dana (mjesec, dan u sedmici, vikend, grijna sezona) i meteorologija.
- **Podjela po vremenu:** trening = ciljni dani 2022–2023, test = cijela 2024. (bez nasumičnog miješanja). Dodatno vremenska unakrsna validacija (`TimeSeriesSplit`) na treningu.
- **Modeli:** baseline (sutra = danas), Ridge / Logistička regresija, Random Forest, HistGradientBoosting. Za klasifikaciju su korišćene težine klasa (`class_weight="balanced"`).
- **Tri skupa feature-a:**
  - **A:** samo zagađenje i kalendar
  - **B:** A + meteorologija do danas (realističan scenarij)
  - **C:** B + meteorologija za sutra, kao da imamo prognozu (optimistična gornja granica)

## Rezultati (test 2024, Random Forest)

| Skup | MAE | MAE, grijna sezona (X–III) | R² | F1 (> 50) | Preciznost | Odziv | AUC |
|---|---|---|---|---|---|---|---|
| Baseline (sutra = danas) | 14,03 | 21,56 | 0,45 | 0,816 | 0,813 | 0,819 | 0,921 |
| A: zagađenje | 13,15 | 19,47 | 0,58 | 0,822 | 0,839 | 0,805 | 0,928 |
| B: + meteo (danas) | 12,41 | 18,44 | 0,63 | 0,823 | 0,872 | 0,779 | 0,938 |
| C: + meteo (sutra) | 11,27 | 16,25 | 0,70 | 0,840 | 0,894 | 0,792 | 0,948 |

MAE je u µg/m³. Svi rezultati su u `results/rezultati_meteo.csv`.

**Glavni nalazi**
- Meteorologija smanjuje grešku: sa 13,1 (samo zagađenje) na 12,4 (B), odnosno 11,3 (C). U grijnoj sezoni, gdje je predviđanje najteže, greška pada sa 21,6 (baseline) na 18,4 (B) i 16,2 (C).
- Najvažniji feature-i su PM2.5 i PM10 iz prethodnih dana, pa minimalna i prosječna temperatura i pritisak (`results/9_znacaj_meteo.png`).
- U klasifikaciji je poboljšanje skromno: F1 sa 0,816 (baseline) na 0,823 (B) i 0,840 (C), a veći dobitak je u preciznosti.

## Ograničenja (važno)

- **Meteorološki podaci nisu mjerenja sa stanice u Valjevu**, već reanaliza ERA5 na mreži ćelija (oko 10–30 km).
- **Skup C je optimističan:** koristi izmjereno vrijeme za sutra umjesto stvarne prognoze. Realan rezultat je između B i C.
- **Test je samo jedna godina**, a trening dvije. Odstupanja u unakrsnoj validaciji su velika (npr. 12,4 ± 5,4), pa razlike među modelima nisu statistički potvrđene.
- **Na drugim stanicama** (Beograd Stari grad, Niš, Beočin) modeli sa samo PM10 prošlošću ne pobjeđuju baseline (`results/rezultati_vise_stanica.csv`). Meteorologija nije testirana na njima jer je preuzeta samo za Valjevo.
- Dio podataka je očišćen ručno: u fajlu za 2022. korišćen je samo prvi blok stanica (državna mreža), a stanice čija su se imena razlikovala između godina nisu uključene. CO za 2022. je poravnat po poziciji jer su datumi u tom listu pomiješani (dan/mjesec).
- Težine klasa mijenjaju balans preciznosti i odziva; prag 0,5 nije optimizovan.
- Nisu testirani SMOTE ni slične metode balansiranja.

## Struktura i pokretanje

```
data/raw/          originalni fajlovi (SEPA, Open-Meteo)
data/processed/    spojeni PM10 po stanicama
src/               skripte (pokreću se redom, iz korijena repozitorija)
results/           grafici i tabele rezultata
```

```bash
pip install -r requirements.txt
python src/01_prepare_data.py
python src/02_eda_valjevo.py
python src/03_models_pollution_only.py
python src/04_multi_station_check.py
python src/05_weather_features.py
```

## Moguća poboljšanja

- Prave meteorološke prognoze (umjesto izmjerenog vremena za sutra) i mjerenja sa najbliže RHMZ stanice.
- Meteorologija za ostale stanice i zajednički model za više stanica.
- Optimizacija praga odlučivanja, SMOTE i kalibracija vjerovatnoća.
- Više godina podataka.
