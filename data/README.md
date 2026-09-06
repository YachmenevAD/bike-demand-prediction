# Data

## Sample data

`sample/bike_rentals.csv` is a small synthetic dataset used for learning
and smoke checks.

## Raw data

The hourly bike rental data is downloaded from the
[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

- Target: `cnt`
- Frequency: hourly
- Period: 2011–2012
- License: CC BY 4.0
- DOI: https://doi.org/10.24432/C5W894

Download the data from the repository root:

```bash
python scripts/download_data.py
```

Downloaded files are stored in `data/raw` and are not tracked by Git.

The columns `casual` and `registered` must not be used as model features
because their sum equals the target:

`cnt = casual + registered`