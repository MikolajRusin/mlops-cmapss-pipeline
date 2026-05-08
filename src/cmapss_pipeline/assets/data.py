import numpy as np
import pandas as pd
import dagster as dg
from pathlib import Path

from ..config import COLUMNS, FEATURES

ROOT_DIR = Path(__file__).parents[3]
DATA_DIR = ROOT_DIR / 'data'


class DatasetConfig(dg.Config):
    dataset: str = 'FD001'

class FeatureEngineeringConfig(dg.Config):
    rul_cap: int = 125
    window: int = 5


def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r'\s+', header=None)
    df = df.dropna(how='all', axis=1)
    df.columns = COLUMNS
    return df

def _add_rolling_features(df: pd.DataFrame, features: list[str], window: int) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        grouped = df.groupby('engine')[col]
        df[f'{col}_roll_mean'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{col}_roll_std'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std(ddof=0).fillna(0)
        )
    return df


@dg.asset
def train_data(config: DatasetConfig) -> pd.DataFrame:
    train_data_path = DATA_DIR / f'train_{config.dataset}.txt'
    df = _load_raw(train_data_path)
    max_cycle = df.groupby('engine')['cycle'].transform('max')
    df['RUL'] = max_cycle - df['cycle']
    return df

@dg.asset
def test_data(config: DatasetConfig) -> dict:
    test_data_path = DATA_DIR / f'test_{config.dataset}.txt'
    rul_data_path  = DATA_DIR / f'RUL_{config.dataset}.txt'
    df  = _load_raw(test_data_path)
    rul = pd.read_csv(rul_data_path, header=None)[0]
    return {'df': df, 'rul': rul}

@dg.asset
def feature_engineering(config: FeatureEngineeringConfig, train_data: pd.DataFrame, test_data: dict) -> dict:
    train_df = _add_rolling_features(train_data, FEATURES, config.window)
    train_df['RUL'] = train_df['RUL'].clip(upper=config.rul_cap)

    test_df = _add_rolling_features(test_data['df'], FEATURES, config.window)
    test_df = test_df.groupby('engine').last().reset_index()
    test_df['RUL'] = test_data['rul'].values

    return {'train': train_df, 'test': test_df}

@dg.asset
def prepare_arrays(feature_engineering: dict) -> dict[str, np.ndarray]:
    train_df = feature_engineering['train']
    test_df  = feature_engineering['test']

    feature_cols = [c for c in train_df.columns if c.endswith(('_roll_mean', '_roll_std'))]

    X_train, y_train = train_df[feature_cols].values, train_df['RUL'].values
    X_test,  y_test  = test_df[feature_cols].values,  test_df['RUL'].values

    return {
        'X_train': X_train,
        'X_test':  X_test,
        'y_train': y_train,
        'y_test':  y_test,
    }
