# DoorDash Delivery Duration Prediction

## Overview

This repository contains a take‑home project for predicting delivery duration (in seconds) for DoorDash orders. The notebook implements two modeling tracks: a direct regression on total delivery time and a two‑stage approach that first predicts preparation time and then reconstructs total duration by adding provided sub‑stage estimates.

Assignment source: fileciteturn3file0

## Problem statement

Given historical order‑level data, predict the total time from order creation (`created_at`) to delivery (`actual_delivery_time`). The dataset includes store, order, and marketplace state features plus two auxiliary predictions available at order time:

* `estimated_order_place_duration`
* `estimated_store_to_consumer_driving_duration`

Target to predict: `total_delivery_time = (actual_delivery_time − created_at)` in seconds. fileciteturn3file0

## Data

Single CSV expected at `datasets/historical_data.csv` with columns described in the brief (time, store, order, marketplace features, and auxiliary predictions). Values are seconds for durations and cents for currency. fileciteturn3file0

## Repository structure

```
.
├── datasets/
│   └── historical_data.csv            # input data (not included)
├── src/
│   └── DeliveryEstimation.ipynb       # main analysis & models
└── README.md
```

## Approach

### 1) Data preparation

* Parsed timestamps and created `total_delivery_time` (seconds).
* Engineered ratios and summaries:

  * `busy_dashers_ratio = total_busy_dashers / total_onshift_dashers` (flagged infinities when on‑shift = 0; dropped rows with inf/NaN afterward).
  * `percent_distinct_item_of_total = num_distinct_items / total_items`
  * `avg_price_per_item = subtotal / total_items`
  * `price_range_of_items = max_item_price − min_item_price`
* One‑hot encoded categorical fields: `order_protocol` and `store_primary_category` (the notebook experimented with `market_id` dummies and later excluded them to avoid over‑fitting to location IDs).
* Removed highly collinear columns after ad‑hoc correlation review; performed a VIF pass to curate a reduced feature set (top ≈40 by importance for tree models).

### 2) Modeling tracks

* Direct regression on `total_delivery_time`

  * Models tried: Ridge, DecisionTreeRegressor (depth‑limited), RandomForestRegressor.
  * Scaling experiments: StandardScaler, MinMaxScaler, and no scaling.
  * Split strategy: random 80/20 train/test via `train_test_split` with `random_state=42`.
* Two‑stage reconstruction of total duration

  * Step A: Predict `prep_time = total_delivery_time − estimated_store_to_consumer_driving_duration − estimated_order_place_duration` using the curated features (scaled with StandardScaler).
  * Step B: Reconstruct and calibrate total duration by combining `prep_time` predictions with the two provided sub‑stage estimates; fit a simple regressor on `[pred_prep, estimated_store_to_consumer_driving_duration, estimated_order_place_duration] → total_delivery_time`.

### 3) Evaluation

* Metric: Root Mean Squared Error (RMSE) on held‑out test data.
* Diagnostics: basic feature importance for Random Forest and bar‑plot summaries of model RMSEs.

## Results (from the notebook)

* Direct regression on total delivery time (full feature set): RMSE ≈ 2048–2057 seconds across models and scalers (Random Forest typically the best among the three in this setup).
* Two‑stage approach (prep‑time → calibrated total):

  * Linear/Ridge regression on the final step achieves RMSE ≈ 1002 seconds on the same random 80/20 split.
  * DecisionTree and RandomForest variants underperform the linear baselines on the final calibration step (e.g., RF RMSE ≈ 1541 s).

## What’s strong

* Sensible decomposition of the problem into sub‑stages, leveraging provided sub‑stage predictions.
* Useful engineered features that normalize across order size and item mix (`avg_price_per_item`, `percent_distinct_item_of_total`, `price_range_of_items`).
* Sanity checks on multicollinearity (VIF) and a feature‑importance pass to constrain the model’s complexity.

## Limitations (plain-language)

1. Train/test split mixes old and new orders
* We randomly split the data. That lets the model learn patterns from the future. Real life moves forward in time, so the score here likely looks better than what we would see in production.

2. Same stores appear in both training and test sets
* If a store is always fast or always slow, and it shows up in both splits, the model gets an easy hint. This can inflate results. A fair test would keep entire stores out of training when they are used for testing.

3. Some inputs may not be known at order time
* We assume every feature is available when the order is created. If any feature is actually known later, the model is using information it would not have in real time.

4. Time effects are missing
* We did not add simple time features like hour of day, day of week, or holidays. These strongly affect delivery time, so the model may miss predictable patterns.

5. What we measure
* We only report RMSE. For ETAs, average absolute error and “how often we are within X minutes” are often more useful. We do not report those.

6. How we treat the target
* We scaled the target directly. With long‑tailed times, using a log transform is often more stable. We did not try that.

7. Handling odd or extreme rows
* When drivers on shift = 0, a ratio can explode. We drop those rows but do not explain or flag them. A safer fix is to cap the ratio and keep a flag.

8. Picking features
* We rely on a quick importance check from one run. That can be unstable. We did not double‑check feature value on truly unseen future data.

9. Comparing models
* We compared models on one random split. Results can change with a different split or when tested on later days.

10. Two‑step design
* In the second approach we predict prep time and then fit another model to combine all parts. In practice, simply adding the parts and using a small correction model would be clearer and less risky.


## How to run

1. Place `historical_data.csv` in `datasets/`.
2. Create a virtual environment and install deps (example):

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
   ```
3. Open the notebook:

   ```bash
   jupyter notebook src/DeliveryEstimation.ipynb
   ```

## License

For educational/portfolio use only. Not affiliated with DoorDash.
