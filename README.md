# PM10 Air Pollution Forecasting for Valjevo (Serbian Open Data)

**Language:** English | [Srpski](README.sr.md)

This project forecasts the **next-day PM10 concentration** (fine particulate matter) and **whether the daily limit of 50 µg/m³ will be exceeded** in Valjevo, Serbia. It uses open air-quality data from the Serbian Environmental Protection Agency (SEPA, via data.gov.rs) for 2022–2024 and weather data (Open-Meteo / ERA5).

Author: Matea Masleša

## Why Valjevo

Of the 16 stations with complete data across all three years, Valjevo shows a clear seasonal pattern (winter mean ≈ 82 µg/m³, summer mean ≈ 27 µg/m³) and many days above the limit (411 of 1,077 measurements, 38%), which gives enough examples for both tasks.

## Data

| Source | Content | File |
|---|---|---|
| SEPA / data.gov.rs | Daily PM10, PM2.5, NO₂, SO₂ and CO values by station, 2022–2024 | `data/raw/2022.xlsx`, `2023.xlsx`, `2024.xlsx` |
| Open-Meteo (ERA5 reanalysis) | Daily temperature, precipitation, wind, pressure, humidity, cloud cover | `data/raw/open-meteo-valjevo-2022-2024.csv` |

`src/01_prepare_data.py` merges PM10 into one table (`data/processed/`) covering the 16 stations present in all three years.
Check the licence of the SEPA datasets on data.gov.rs before reusing them. Weather data: Open-Meteo.com, licence CC BY 4.0 (source: Copernicus ERA5).

## Methodology

- **Targets:** next-day PM10 (regression) and next-day PM10 > 50 µg/m³ (classification).
- **Features** (only information available up to today): PM10 from previous days (lags), rolling means and maximum, PM2.5, NO₂, CO, calendar of the target day (month, weekday, weekend, heating season) and weather.
- **Time-based split:** training = target days in 2022–2023, test = all of 2024 (no random shuffling). Additionally, time-series cross-validation (`TimeSeriesSplit`) on the training set.
- **Models:** baseline (tomorrow = today), Ridge / Logistic Regression, Random Forest, HistGradientBoosting. Class weights (`class_weight="balanced"`) are used for classification.
- **Three feature sets:**
  - **A:** pollution and calendar only
  - **B:** A + weather up to today (realistic scenario)
  - **C:** B + weather for tomorrow, as if a forecast were available (optimistic upper bound)

## Results (test 2024, Random Forest)

| Feature set | MAE | MAE, heating season (Oct–Mar) | R² | F1 (> 50) | Precision | Recall | AUC |
|---|---|---|---|---|---|---|---|
| Baseline (tomorrow = today) | 14.03 | 21.56 | 0.45 | 0.816 | 0.813 | 0.819 | 0.921 |
| A: pollution | 13.15 | 19.47 | 0.58 | 0.822 | 0.839 | 0.805 | 0.928 |
| B: + weather (today) | 12.41 | 18.44 | 0.63 | 0.823 | 0.872 | 0.779 | 0.938 |
| C: + weather (tomorrow) | 11.27 | 16.25 | 0.70 | 0.840 | 0.894 | 0.792 | 0.948 |

MAE is in µg/m³. All results are in `results/rezultati_meteo.csv`.

**Key findings**
- Weather reduces the error: from 13.1 (pollution only) to 12.4 (B) and 11.3 (C). In the heating season, where forecasting is hardest, the error drops from 21.6 (baseline) to 18.4 (B) and 16.2 (C).
- The most important features are PM2.5 and PM10 from previous days, followed by minimum and mean temperature and pressure (`results/9_znacaj_meteo.png`).
- In classification the improvement is modest: F1 rises from 0.816 (baseline) to 0.823 (B) and 0.840 (C), with the larger gain in precision.

## Limitations (important)

- **The weather data are not measurements from a station in Valjevo** but ERA5 reanalysis on a grid (roughly 10–30 km resolution).
- **Set C is optimistic:** it uses observed weather for tomorrow instead of a real forecast. The realistic result lies between B and C.
- **The test set is a single year**, with two years of training. Cross-validation variance is large (e.g. 12.4 ± 5.4), so differences between models are not statistically confirmed.
- **On other stations** (Belgrade Stari grad, Niš, Beočin) models using only past PM10 do not beat the baseline (`results/rezultati_vise_stanica.csv`). Weather was not tested there because it was downloaded for Valjevo only.
- Some data were cleaned manually: for 2022 only the first block of stations (the national network) was used, and stations whose names differed between years were excluded. 2022 CO was aligned by position because the dates in that sheet were mixed up (day/month).
- Class weights shift the balance between precision and recall; the 0.5 threshold was not optimised.
- SMOTE and similar resampling methods were not tested.

## Structure and usage

```
data/raw/          original files (SEPA, Open-Meteo)
data/processed/    merged PM10 by station
src/               scripts (run in order, from the repository root)
results/           figures and result tables
```

```bash
pip install -r requirements.txt
python src/01_prepare_data.py
python src/02_eda_valjevo.py
python src/03_models_pollution_only.py
python src/04_multi_station_check.py
python src/05_weather_features.py
```

## Possible improvements

- Real weather forecasts (instead of observed weather for tomorrow) and measurements from the nearest RHMZ station.
- Weather for the other stations and a joint multi-station model.
- Threshold optimisation, SMOTE and probability calibration.
- More years of data.
