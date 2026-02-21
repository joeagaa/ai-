from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    data_dir: str = "data"
    train_csv: str = "data/train.csv"
    out_dir: str = "out"
    models_dir: str = "out/models"
    logs_dir: str = "out/logs"

@dataclass(frozen=True)
class Settings:
    test_size: float = 0.2
    random_state: int = 42

PATHS = Paths()
SETTINGS = Settings()
