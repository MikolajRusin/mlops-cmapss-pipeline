import numpy as np
import pandas as pd
import dagster as dg
from pathlib import Path

from ..config import COLUMNS, FEATURES

ROOT_DIR = Path(__file__).parents[3]
DATA_DIR = ROOT_DIR / 'data'


class DatasetConfig(dg.Config):
    dataset: str = 'FD001'

def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r'\s+', header=None)
    df = df.dropna(how='all', axis=1)
    df.columns = COLUMNS
    return df

@dg.asset
def train_data(config: DatasetConfig) -> pd.DataFrame:
    train_data_path = DATA_DIR / f'train_{config.dataset}.txt'
    df = _load_raw(train_data_path)
    max_cycle = df.groupby('engine')['cycle'].transform('max')
    df['RUL'] = max_cycle - df['cycle']
    return df

@dg.asset
def test_data(config: DatasetConfig) -> pd.DataFrame:
    test_data_path = DATA_DIR / f'test_{config.dataset}.txt'
    rul_data_path  = DATA_DIR / f'RUL_{config.dataset}.txt'
    df = _load_raw(test_data_path)
    df = df.groupby('engine').last().reset_index()
    rul = pd.read_csv(rul_data_path, header=None)[0]
    df['RUL'] = rul
    return df

@dg.asset
def split_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict[str, np.ndarray]:
    X_train, y_train = train_data[FEATURES].values, train_data['RUL'].values
    X_test, y_test = test_data[FEATURES].values, test_data['RUL'].values
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }