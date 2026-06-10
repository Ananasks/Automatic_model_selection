"""Automatic model selection for ionospheric time-series regression.

The script keeps the whole research pipeline in one place:
data loading, time-feature engineering, chronological validation,
model comparison, artifact saving, and fine-tuning on later intervals.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


ROWS_PER_DAY = 48
TEST_SIZE = 0.2
HP_FILTER_LAMBDA = 10
RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one target variable and one training interval."""

    file_path: str
    target_column: str
    day_start: int
    day_stop: int
    dataset_name: str
    models_dir: str = "trained_models"

    @property
    def model_dir(self) -> Path:
        return Path(self.models_dir) / self.dataset_name


def print_header(title: str, symbol: str = "=") -> None:
    print("\n" + symbol * 80)
    print(title)
    print(symbol * 80)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features that help classical models work with time series."""

    df = df.copy()
    df["datetime"] = df.index
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["quarter"] = df["datetime"].dt.quarter
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["day"] = df["datetime"].dt.day
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
    return df


def build_datetime_column(df: pd.DataFrame) -> pd.Series:
    """Support both CSV layouts used in the project datasets."""

    if "datetime" in df.columns and "date" not in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")

    if "date" in df.columns and "time" in df.columns:
        date_time = df["date"].astype(str) + " " + df["time"].astype(str)
        return pd.to_datetime(date_time, errors="coerce")

    raise ValueError("Нужна колонка 'datetime' или пара колонок 'date' + 'time'.")


def load_time_series(file_path: str, target_column: str) -> pd.DataFrame:
    """Load a CSV file, create a datetime index, add features, and clean rows."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path)
    df["datetime"] = build_datetime_column(df)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = add_time_features(df)
    df = df.drop(columns=["index", "time", "date"], errors="ignore")

    if target_column not in df.columns:
        raise KeyError(f"Целевая колонка '{target_column}' не найдена в {path}")

    # HP-filter keeps the slower trend component of the target signal.
    try:
        _, df[target_column] = sm.tsa.filters.hpfilter(df[target_column], HP_FILTER_LAMBDA)
    except Exception as error:
        print(f"Предупреждение: HP-фильтр для '{target_column}' не применен: {error}")

    return df.dropna()


def make_feature_target_split(
    df: pd.DataFrame,
    target_column: str,
    day_start: int,
    day_stop: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select a day range and split it into features and target values."""

    if day_stop <= day_start:
        raise ValueError("day_stop должен быть больше day_start")

    start_row = day_start * ROWS_PER_DAY
    stop_row = day_stop * ROWS_PER_DAY
    selected_days = df.iloc[start_row:stop_row]

    if selected_days.empty:
        raise ValueError(
            f"Нет данных для диапазона дней {day_start}-{day_stop}. "
            f"Доступно строк после обработки: {len(df)}"
        )

    X = selected_days.drop(columns=[target_column, "datetime"], errors="ignore")
    y = selected_days[target_column].copy()

    if X.empty:
        raise ValueError(f"Нет признаков после удаления целевой колонки '{target_column}'")
    if len(X) < 5:
        raise ValueError(f"Недостаточно строк для обучения и проверки: {len(X)}")

    return X, y


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split without shuffling, so the test set is always later in time."""

    test_rows = max(1, int(round(len(X) * test_size)))
    train_rows = len(X) - test_rows

    if train_rows < 1:
        raise ValueError("Недостаточно данных: train-часть получилась пустой")

    X_train = X.iloc[:train_rows]
    X_test = X.iloc[train_rows:]
    y_train = y.iloc[:train_rows]
    y_test = y.iloc[train_rows:]
    return X_train, X_test, y_train, y_test


def scale_features(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fit_scaler: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale features with train-only fitting to avoid data leakage."""

    if fit_scaler:
        train_values = scaler.fit_transform(X_train)
    else:
        train_values = scaler.transform(X_train)

    test_values = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(train_values, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(test_values, columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled


def evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate the metrics used for model comparison and reports."""

    predictions = model.predict(X_test)
    return {
        "r2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "MAPE": mean_absolute_percentage_error(y_test, predictions),
        "predictions": predictions,
    }


def print_metrics(metrics: dict) -> None:
    print(f"    R2 = {metrics['r2']:.4f}")
    print(f"    MAE = {metrics['MAE']:.4f}")
    print(f"    MAPE = {metrics['MAPE']:.4f}")


class ModelTrainer:
    """Train candidate models and save the best one with preprocessing artifacts."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_dir = config.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.df = load_time_series(config.file_path, config.target_column)
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model_name = None

    @staticmethod
    def create_models() -> dict:
        """Return a compact baseline set: linear, regularized linear, and MLP."""

        return {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(alpha=0.1, max_iter=5000, random_state=RANDOM_STATE),
            "MLPRegressor": MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        }

    def save_model(self, model, model_name: str) -> None:
        model_path = self.model_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"  ok: {model_name} сохранена: {model_path}")

    def save_training_artifacts(self, feature_names: list[str]) -> None:
        joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
        joblib.dump(feature_names, self.model_dir / "feature_names.pkl")
        print(f"  ok: scaler и {len(feature_names)} признаков сохранены")

    def select_best_model(self, results: dict) -> str:
        """Pick the model with the highest R2 score and persist the decision."""

        best_model_name = max(results, key=lambda name: results[name]["r2"])
        self.best_model_name = best_model_name

        best_metrics = results[best_model_name]
        best_info = {
            "best_model": best_model_name,
            "target_column": self.config.target_column,
            "r2": best_metrics["r2"],
            "mae": best_metrics["MAE"],
            "mape": best_metrics["MAPE"],
        }
        pd.DataFrame([best_info]).to_excel(self.model_dir / "best_model_info.xlsx", index=False)
        return best_model_name

    def save_results_table(self, results: dict) -> None:
        rows = [
            {
                "model": model_name,
                "r2": metrics["r2"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
            }
            for model_name, metrics in results.items()
        ]
        output_path = self.model_dir / f"training_results_{self.config.target_column}.xlsx"
        pd.DataFrame(rows).to_excel(output_path, index=False)
        print(f"\nok: результаты сохранены в {output_path}")

    def run(self) -> tuple[dict, str]:
        print_header(f"ОБУЧЕНИЕ: {self.config.dataset_name} | {self.config.target_column}")

        X, y = make_feature_target_split(
            self.df,
            self.config.target_column,
            self.config.day_start,
            self.config.day_stop,
        )
        X_train_raw, X_test_raw, y_train, y_test = chronological_split(X, y)

        print("\nНормализация признаков...")
        X_train, X_test = scale_features(self.scaler, X_train_raw, X_test_raw, fit_scaler=True)
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        results = {}
        print("\nОбучение моделей...")
        for model_name, model in self.create_models().items():
            print(f"\n  Обучение {model_name}...")
            model.fit(X_train, y_train)
            metrics = evaluate_regression(model, X_test, y_test)
            print_metrics(metrics)

            results[model_name] = metrics
            self.models[model_name] = model
            self.save_model(model, model_name)

        self.save_training_artifacts(list(X.columns))

        print_header("АВТОМАТИЧЕСКИЙ ВЫБОР ЛУЧШЕЙ МОДЕЛИ")
        best_model_name = self.select_best_model(results)
        print(f"\nok: лучшая модель: {best_model_name}")
        print_metrics(results[best_model_name])

        self.save_results_table(results)
        return results, best_model_name


class ModelFineTuner:
    """Reload the selected model and adapt it to a new interval of the same target."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_dir = config.model_dir
        self.df = load_time_series(config.file_path, config.target_column)

        self.scaler = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None

    def load_artifacts(self) -> None:
        print(f"\nЗагрузка артефактов из {self.model_dir}...")

        scaler_path = self.model_dir / "scaler.pkl"
        features_path = self.model_dir / "feature_names.pkl"
        best_info_path = self.model_dir / "best_model_info.xlsx"

        for path in [scaler_path, features_path, best_info_path]:
            if not path.exists():
                raise FileNotFoundError(f"Не найден обязательный артефакт: {path}")

        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)

        best_info = pd.read_excel(best_info_path)
        self.best_model_name = best_info.loc[0, "best_model"]

        model_path = self.model_dir / f"{self.best_model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.best_model = joblib.load(model_path)
        print(f"  ok: модель {self.best_model_name} загружена")
        print(f"  ok: загружено признаков: {len(self.feature_names)}")

    def align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Match new data to the exact feature set used during initial training."""

        X = X.copy()

        for column in self.feature_names:
            if column not in X.columns:
                X[column] = 0

        extra_columns = [column for column in X.columns if column not in self.feature_names]
        if extra_columns:
            print(f"  info: лишние признаки отброшены: {len(extra_columns)}")

        return X[self.feature_names]

    def fine_tune_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the saved model on a new chronological training interval."""

        print(f"\nДообучение модели {self.best_model_name}...")

        if isinstance(self.best_model, MLPRegressor):
            self.best_model.warm_start = True

        self.best_model.fit(X_train, y_train)
        return self.best_model

    def save_finetuned_model(self) -> None:
        model_path = self.model_dir / f"{self.best_model_name}_finetuned_{self.config.target_column}.pkl"
        joblib.dump(self.best_model, model_path)
        print(f"  ok: дообученная модель сохранена: {model_path}")

    def save_results_table(self, metrics: dict) -> None:
        output_path = (
            self.model_dir
            / f"finetuning_results_{self.config.target_column}_days_{self.config.day_start}_{self.config.day_stop}.xlsx"
        )
        row = {
            "target_column": self.config.target_column,
            "model": self.best_model_name,
            "day_start": self.config.day_start,
            "day_stop": self.config.day_stop,
            "r2": metrics["r2"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
        }
        pd.DataFrame([row]).to_excel(output_path, index=False)
        print(f"\nok: результаты сохранены в {output_path}")

    def run(self) -> dict | None:
        print_header(
            f"ДООБУЧЕНИЕ: {self.config.dataset_name} | {self.config.target_column} | "
            f"дни {self.config.day_start}-{self.config.day_stop}"
        )

        self.load_artifacts()

        X, y = make_feature_target_split(
            self.df,
            self.config.target_column,
            self.config.day_start,
            self.config.day_stop,
        )
        print(f"\nРазмер данных для дообучения: {len(X)}")

        X = self.align_features(X)
        X_train_raw, X_test_raw, y_train, y_test = chronological_split(X, y)
        X_train, X_test = scale_features(self.scaler, X_train_raw, X_test_raw, fit_scaler=False)

        print(f"Используем {len(X.columns)} признаков")
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        self.fine_tune_model(X_train, y_train)

        print("\nОценка дообученной модели:")
        metrics = evaluate_regression(self.best_model, X_test, y_test)
        print_metrics(metrics)

        self.save_finetuned_model()
        self.save_results_table(metrics)
        return metrics


def train_experiment(config: ExperimentConfig) -> str:
    """Run one initial training experiment and return the selected model name."""

    _, best_model_name = ModelTrainer(config).run()
    return best_model_name


def run_finetune_tests(base_config: ExperimentConfig, test_ranges: list[dict]) -> None:
    """Run the same fine-tuning scenario across several day ranges."""

    for index, test_range in enumerate(test_ranges, start=1):
        description = test_range["description"]
        day_start = test_range["day_start"]
        day_stop = test_range["day_stop"]

        print(f"\n--- Тест {index}: {description}, дни {day_start}-{day_stop} ---")
        config = ExperimentConfig(
            file_path=base_config.file_path,
            target_column=base_config.target_column,
            day_start=day_start,
            day_stop=day_stop,
            dataset_name=base_config.dataset_name,
            models_dir=base_config.models_dir,
        )
        ModelFineTuner(config).run()


def main() -> None:
    """Run the portfolio demo pipeline on the configured datasets."""

    abs_file = "absTEC_absCB_dayStart_1_daysCount_309_year_2022_startStation_tetu_filted_CB.csv"
    muf_file = "total_muf_30_minuts.csv"

    experiments = {
        "absTEC": ExperimentConfig(abs_file, "kabc_absTEC", 177, 210, "absTEC"),
        "absCB": ExperimentConfig(abs_file, "kabc_absCB", 177, 210, "absCB"),
        "MUF": ExperimentConfig(muf_file, "muf", 50, 100, "MUF"),
    }

    print_header("ЭТАП 1: ОБУЧЕНИЕ МОДЕЛЕЙ", "#")
    best_models = {
        name: train_experiment(config)
        for name, config in experiments.items()
    }

    print_header("ЭТАП 2: ДООБУЧЕНИЕ НА НОВЫХ УЧАСТКАХ", "#")
    run_finetune_tests(
        experiments["absCB"],
        [
            {"day_start": 177, "day_stop": 210, "description": "absCB участок 1"},
            {"day_start": 50, "day_stop": 100, "description": "absCB участок 2"},
            {"day_start": 100, "day_stop": 150, "description": "absCB участок 3"},
        ],
    )
    run_finetune_tests(
        experiments["MUF"],
        [
            {"day_start": 10, "day_stop": 20, "description": "MUF участок 1"},
            {"day_start": 15, "day_stop": 30, "description": "MUF участок 2"},
            {"day_start": 5, "day_stop": 40, "description": "MUF участок 3"},
        ],
    )

    print_header("ОБУЧЕНИЕ ЗАВЕРШЕНО", "#")
    for dataset_name, model_name in best_models.items():
        print(f"ok: лучшая модель для {dataset_name}: {model_name}")
    print("ok: проведено тестов дообучения: 6")
    print("ok: результаты сохранены в директории 'trained_models/'")


if __name__ == "__main__":
    main()
