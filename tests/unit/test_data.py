import pytest
import numpy as np
from pathlib import Path

from src.cmapss_pipeline.config import COLUMNS, FEATURES
from src.cmapss_pipeline.assets.data import (
    _load_raw,
    train_data, test_data as cmapss_test_data, DatasetConfig,
    feature_engineering, FeatureEngineeringConfig,
    prepare_arrays,
)

ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / 'data'


@pytest.fixture
def dataset_config():
    return DatasetConfig(dataset='FD001')

@pytest.fixture
def fe_config():
    return FeatureEngineeringConfig(rul_cap=125, window=5)

@pytest.fixture
def cmapss_train_raw_df():
    return _load_raw(DATA_DIR / 'train_FD001.txt')

@pytest.fixture
def cmapss_test_raw_df():
    return _load_raw(DATA_DIR / 'test_FD001.txt')

@pytest.fixture
def cmapss_train_df(dataset_config):
    return train_data(dataset_config)

@pytest.fixture
def cmapss_test_dict(dataset_config):
    return cmapss_test_data(dataset_config)

@pytest.fixture
def cmapss_feature_engineering(fe_config, cmapss_train_df, cmapss_test_dict):
    return feature_engineering(fe_config, cmapss_train_df, cmapss_test_dict)


# ===== _load_raw tests =====
def test_load_raw_shape(cmapss_train_raw_df, cmapss_test_raw_df):
    assert cmapss_train_raw_df.shape[0] > 0
    assert cmapss_test_raw_df.shape[0] > 0
    assert cmapss_train_raw_df.shape[1] == len(COLUMNS)
    assert cmapss_test_raw_df.shape[1] == len(COLUMNS)

def test_load_raw_columns(cmapss_train_raw_df, cmapss_test_raw_df):
    assert list(cmapss_train_raw_df.columns) == COLUMNS
    assert list(cmapss_test_raw_df.columns) == COLUMNS

def test_load_raw_no_empty_cols(cmapss_train_raw_df, cmapss_test_raw_df):
    assert not cmapss_train_raw_df.isna().all().any()
    assert not cmapss_test_raw_df.isna().all().any()


# ===== train_data test =====
def test_train_data_has_rul(cmapss_train_df):
    assert 'RUL' in cmapss_train_df.columns

def test_train_data_rul_min_zero(cmapss_train_df):
    assert cmapss_train_df['RUL'].min() == 0


# ===== test_data tests =====
def test_test_data_returns_dict(cmapss_test_dict):
    assert isinstance(cmapss_test_dict, dict)
    assert 'df' in cmapss_test_dict
    assert 'rul' in cmapss_test_dict

def test_test_data_df_columns(cmapss_test_dict):
    assert list(cmapss_test_dict['df'].columns) == COLUMNS

def test_test_data_rul_gte_zero(cmapss_test_dict):
    assert cmapss_test_dict['rul'].min() >= 0


# ===== feature_engineering tests =====
def test_feature_engineering_keys(cmapss_feature_engineering):
    assert 'train' in cmapss_feature_engineering
    assert 'test' in cmapss_feature_engineering

def test_feature_engineering_one_row_per_engine(cmapss_feature_engineering):
    test_df = cmapss_feature_engineering['test']
    assert test_df['engine'].nunique() == len(test_df)

def test_feature_engineering_train_keeps_all_cycles(cmapss_feature_engineering):
    train_df = cmapss_feature_engineering['train']
    assert len(train_df) > train_df['engine'].nunique()

def test_feature_engineering_rolling_cols_exist(cmapss_feature_engineering):
    train_df       = cmapss_feature_engineering['train']
    roll_mean_cols = [c for c in train_df.columns if c.endswith('_roll_mean')]
    roll_std_cols  = [c for c in train_df.columns if c.endswith('_roll_std')]
    assert len(roll_mean_cols) == len(FEATURES)
    assert len(roll_std_cols)  == len(FEATURES)

def test_feature_engineering_rul_cap(cmapss_feature_engineering):
    assert cmapss_feature_engineering['train']['RUL'].max() <= 125

def test_feature_engineering_test_has_rul(cmapss_feature_engineering):
    assert 'RUL' in cmapss_feature_engineering['test'].columns


# ===== prepare_arrays tests =====
def test_prepare_arrays_len(cmapss_feature_engineering):
    assert len(prepare_arrays(cmapss_feature_engineering)) == 4

def test_prepare_arrays_keys(cmapss_feature_engineering):
    result = prepare_arrays(cmapss_feature_engineering)
    assert list(result.keys()) == ['X_train', 'X_test', 'y_train', 'y_test']

def test_prepare_arrays_values_are_ndarray(cmapss_feature_engineering):
    result = prepare_arrays(cmapss_feature_engineering)
    assert all(isinstance(v, np.ndarray) for v in result.values())

def test_prepare_arrays_feature_count(cmapss_feature_engineering):
    result         = prepare_arrays(cmapss_feature_engineering)
    expected_cols  = len(FEATURES) * 2  # roll_mean + roll_std
    assert result['X_train'].shape[1] == expected_cols
    assert result['X_test'].shape[1]  == expected_cols

def test_prepare_arrays_xy_consistent(cmapss_feature_engineering):
    result = prepare_arrays(cmapss_feature_engineering)
    assert len(result['X_train']) == len(result['y_train'])
    assert len(result['X_test'])  == len(result['y_test'])
