from statistics import mean, stdev
from typing import Any

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.data import TEST_START, load_hourly_data
from src.real_features import select_features_and_target
from src.train_real_models import (
    build_linear_pipeline,
    build_random_forest_pipeline,
)


N_SPLITS = 3
VALIDATION_WINDOW_SIZE = 24 * 30


def main() -> None:
    data = load_hourly_data()

    # Test data must not participate in model selection.
    development_data = data.loc[
        data["timestamp"] < TEST_START
    ].copy()
    development_data = development_data.reset_index(drop=True)

    X, y = select_features_and_target(development_data)

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_WINDOW_SIZE,
    )

    models: dict[str, Any] = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": build_linear_pipeline(),
        "RandomForestRegressor": build_random_forest_pipeline(),
    }

    scores: dict[str, list[float]] = {
        name: []
        for name in models
    }

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_fold_train = X.iloc[train_indices]
        y_fold_train = y.iloc[train_indices]

        X_fold_validation = X.iloc[validation_indices]
        y_fold_validation = y.iloc[validation_indices]

        validation_start = development_data.iloc[
            validation_indices[0]
        ]["timestamp"]
        validation_end = development_data.iloc[
            validation_indices[-1]
        ]["timestamp"]

        print(
            f"\nFold {fold_number}: "
            f"{len(train_indices)} train rows, "
            f"{len(validation_indices)} validation rows"
        )
        print(
            f"Validation period: "
            f"{validation_start} -> {validation_end}"
        )

        for model_name, model in models.items():
            fold_model = clone(model)
            fold_model.fit(X_fold_train,y_fold_train)
            predictions = fold_model.predict(X_fold_validation)
            fold_mae = mean_absolute_error(
                y_fold_validation,
                predictions,
            )

            scores[model_name].append(float(fold_mae))
            print(f"{model_name} MAE: {fold_mae:.2f}")

    print("\nCross-validation summary:")

    for model_name, model_scores in scores.items():
        mean_mae = mean(model_scores)
        std_mae = stdev(model_scores)

        print(
            f"{model_name}: "
            f"mean MAE = {mean_mae:.2f}, "
            f"std = {std_mae:.2f}, "
            f"folds = {[round(value, 2) for value in model_scores]}"
        )


if __name__ == "__main__":
    main()