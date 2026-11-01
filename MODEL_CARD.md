# Model Card: Bike Demand Random Forest

## Model details

- **Task:** hourly bike rental demand regression
- **Model:** `RandomForestRegressor`
- **Model version:** 1.0
- **Framework:** scikit-learn
- **Artifact format:** joblib
- **Target:** `cnt`, the total number of rentals during an hour

The serialized artifact contains the complete scikit-learn pipeline, including
feature preprocessing and the trained regression model.

## Intended use

The model is intended for:

- demonstrating an end-to-end ML engineering workflow;
- estimating hourly demand on data compatible with the UCI dataset schema;
- comparing demand across combinations of time and weather conditions;
- educational and portfolio purposes.

The model is not intended for:

- direct use in a current production bike-sharing system;
- safety-critical or fully automated operational decisions;
- causal analysis of factors affecting demand;
- forecasting demand in other cities without retraining and validation;
- making decisions about individual users.

## Training data

The project uses the
[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

The dataset contains 17,379 hourly observations from 2011 and 2012.

The final evaluated model was trained on 15,211 observations from:

```text
2011-01-01 00:00:00 — 2012-09-30 23:00:00
```

The held-out test set contains 2,168 observations from:

```text
2012-10-01 00:00:00 — 2012-12-31 23:00:00
```

The data was split chronologically to prevent future observations from
appearing in the training data.

## Features

Categorical features:

- `season`;
- `mnth`;
- `hr`;
- `weekday`;
- `weathersit`.

Binary features:

- `yr`;
- `holiday`;
- `workingday`.

Numeric features:

- `temp`;
- `atemp`;
- `hum`;
- `windspeed`.

The weather measurements are normalized values defined by the original
dataset. The API therefore expects the same representation rather than raw
physical units.

The following columns are excluded:

- `casual`;
- `registered`;
- `cnt`;
- `instant`;
- date and timestamp columns.

The columns `casual` and `registered` are forbidden model features because:

```text
cnt = casual + registered
```

Using them would cause target leakage.

## Preprocessing

The model uses a scikit-learn `Pipeline` with a `ColumnTransformer`:

- categorical features are encoded with `OneHotEncoder`;
- binary features are passed through unchanged;
- numeric features are passed through unchanged;
- the transformed features are provided to `RandomForestRegressor`.

The complete pipeline is saved to prevent training-serving skew.

## Training and model selection

The following model types were compared:

- `DummyRegressor`;
- `LinearRegression`;
- `RandomForestRegressor`.

Model selection used expanding-window temporal cross-validation with three
validation folds of approximately 30 days each.

Selected Random Forest parameters:

```text
n_estimators = 150
min_samples_leaf = 1
max_depth = None
random_state = 42
```

## Evaluation results

Final metrics on the independent test period:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| DummyRegressor | 156.87 | 204.43 | -0.03 |
| LinearRegression | 89.69 | 121.46 | 0.64 |
| **RandomForestRegressor** | **53.81** | **81.29** | **0.84** |

The selected model reduced test MAE by approximately 65.7% relative to the
constant-mean baseline.

## Error analysis

Validation error analysis showed:

- a mean signed error of `-16.92`, indicating average underprediction;
- the highest absolute errors occurred around morning and evening rush hours;
- the highest weather-group MAE occurred for weather category `3`;
- category `3` contained substantially fewer observations than clear weather,
  so its error estimate is less stable.

These results show association rather than causation. Grouped errors may also
be affected by seasonality, demand scale, weather, and other confounding
factors.

## Limitations and risks

- The data is from 2011–2012 and may not represent current demand.
- The dataset describes one bike-sharing system and does not establish
  generalization to other cities.
- Random Forest models cannot reliably extrapolate beyond patterns observed
  during training.
- Rush-hour demand has substantially larger absolute errors.
- Rare and severe weather conditions have limited representation.
- The model produces a point estimate without an uncertainty interval.
- The API validates individual field ranges but does not yet validate all
  relationships between fields. For example, contradictory month and season
  values can be submitted.
- The `yr` field is specific to the original dataset and only supports the
  encoded values `0` and `1`.
- The model artifact is approximately 188 MB.
- Production drift monitoring and automated retraining are not implemented.
- The service does not implement authentication or rate limiting.

## Ethical considerations

The dataset contains aggregate hourly counts and does not include personal
identifiers or protected user attributes.

However, operational decisions based on inaccurate demand forecasts could
lead to unequal bike availability between locations. Deployment in a real
system would require location-level evaluation, monitoring, and human
oversight.

## Security considerations

Joblib artifacts must only be loaded from trusted sources. Loading a malicious
pickle-compatible artifact can execute arbitrary Python code.

The Docker container mounts the model artifact as read-only to prevent the
serving process from modifying it.

## Reproducibility

Download the data and train the model from the repository root:

```bash
python scripts/download_data.py
python -m src.train
```

Generated files:

```text
artifacts/model.joblib
artifacts/metrics.json
```

The repository also records model parameters, dependencies, data splitting
logic, tests, and CI configuration.