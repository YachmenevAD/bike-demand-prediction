from statistics import mean, stdev

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.data import TEST_START, load_hourly_data
from src.real_features import select_features_and_target
from src.train_real_models import build_random_forest_pipeline


N_SPLITS = 3
VALIDATION_WINDOW_SIZE = 24 * 30
N_ESTIMATORS = 150

CANDIDATES: list[tuple[str, int, int | None]] = [
    ("leaf_1_unlimited", 1, None),
    ("leaf_2_unlimited", 2, None),
    ("leaf_5_unlimited", 5, None),
    ("leaf_2_depth_20", 2, 20),
]


def main() -> None:
    data = load_hourly_data()

    development_data = data.loc[
        data["timestamp"] < TEST_START
    ].copy()
    development_data = development_data.reset_index(drop=True)

    X, y = select_features_and_target(development_data)

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_WINDOW_SIZE,
    )

    # All candidates must be evaluated on exactly the same folds.
    folds = list(splitter.split(X))

    results: dict[str, tuple[float, float]] = {}

    for candidate_name, min_samples_leaf, max_depth in CANDIDATES:
        fold_scores: list[float] = []

        print(f"\nCandidate: {candidate_name}")
        print(f"min_samples_leaf: {min_samples_leaf}")
        print(f"max_depth: {max_depth}")

        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(folds, start=1):
            X_fold_train = X.iloc[train_indices]
            y_fold_train = y.iloc[train_indices]

            X_fold_validation = X.iloc[validation_indices]
            y_fold_validation = y.iloc[validation_indices]

            model = build_random_forest_pipeline()

            model.set_params(
                model__n_estimators=N_ESTIMATORS,
                model__min_samples_leaf=min_samples_leaf,
                model__max_depth=max_depth,
            )
            model.fit(X_fold_train,y_fold_train)
            predictions = model.predict(X_fold_validation)
            fold_mae = mean_absolute_error(
                y_fold_validation,
                predictions
            )

            fold_scores.append(float(fold_mae))
            print(f"Fold {fold_number} MAE: {fold_mae:.2f}")

        mean_mae = mean(fold_scores)
        std_mae = stdev(fold_scores)

        results[candidate_name] = (mean_mae, std_mae)

        print(
            f"Summary: mean MAE = {mean_mae:.2f}, "
            f"std = {std_mae:.2f}"
        )

    ranked_results = sorted(
        results.items(),
        key=lambda item: item[1][0],
    )

    print("\nCandidate ranking:")

    for position, (
        candidate_name,
        (mean_mae, std_mae),
    ) in enumerate(ranked_results, start=1):
        print(
            f"{position}. {candidate_name}: "
            f"mean MAE = {mean_mae:.2f}, "
            f"std = {std_mae:.2f}"
        )

    best_name, (best_mean_mae, best_std_mae) = ranked_results[0]

    print(
        "\nBest candidate: "
        f"{best_name} "
        f"(mean MAE = {best_mean_mae:.2f}, "
        f"std = {best_std_mae:.2f})"
    )


if __name__ == "__main__":
    main()