import pytest
import numpy as np
from pathlib import Path

from src.cmapss_pipeline.config import COLUMNS, FEATURES
from src.cmapss_pipeline.assets.data import (
    _load_raw, train_data, test_data as cmapss_test_data, DatasetConfig,
    split_data
)

ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / 'data'


@pytest.fixture
def cmapss_train_raw_df():
    train_path = DATA_DIR / 'train_FD001.txt'
    return _load_raw(train_path)

@pytest.fixture
def cmapss_test_raw_df():
    test_path = DATA_DIR / 'test_FD001.txt'
    return _load_raw(test_path)

@pytest.fixture
def dataset_config():
    return DatasetConfig(dataset='FD001')

@pytest.fixture
def cmapss_train_df(dataset_config):
    return train_data(dataset_config)

@pytest.fixture
def cmapss_test_df(dataset_config):
    return cmapss_test_data(dataset_config)


# ===== _load_raw tests for both train and test paths =====
def test_load_raw_shape(cmapss_train_raw_df, cmapss_test_raw_df):
    train_rows, train_cols = cmapss_train_raw_df.shape
    test_rows, test_cols = cmapss_test_raw_df.shape
    assert train_rows > 0
    assert test_rows > 0
    assert train_cols == len(COLUMNS)
    assert test_cols == len(COLUMNS)

def test_load_raw_columns(cmapss_train_raw_df, cmapss_test_raw_df):
    assert all(cmapss_train_raw_df.columns == COLUMNS)
    assert all(cmapss_test_raw_df.columns == COLUMNS)

def test_load_raw_no_empty_cols(cmapss_train_raw_df, cmapss_test_raw_df):
    assert not cmapss_train_raw_df.isna().all().any()
    assert not cmapss_test_raw_df.isna().all().any()


# ===== train_data test =====
def test_train_data_has_rul(cmapss_train_df):
    assert 'RUL' in cmapss_train_df.columns

def test_train_data_rul_min_zero(cmapss_train_df):
    assert cmapss_train_df['RUL'].min() == 0


# ===== test_data tests =====
def test_test_data_has_rul(cmapss_test_df):
    assert 'RUL' in cmapss_test_df.columns

def test_test_data_rul_gte_zero(cmapss_test_df):
    assert cmapss_test_df['RUL'].min() >= 0

def test_test_data_one_row_per_engine(cmapss_test_df):
    assert cmapss_test_df['engine'].nunique() == len(cmapss_test_df)


# ===== split_data tests =====
def test_split_data_len(cmapss_train_df, cmapss_test_df):
    splitted_data = split_data(cmapss_train_df, cmapss_test_df)
    assert len(splitted_data) == 4

def test_split_data_keys(cmapss_train_df, cmapss_test_df):
    splitted_data = split_data(cmapss_train_df, cmapss_test_df)
    assert ['X_train', 'X_test', 'y_train', 'y_test'] == list(splitted_data.keys())

def test_split_data_values_are_ndarray(cmapss_train_df, cmapss_test_df):
    splitted_data = split_data(cmapss_train_df, cmapss_test_df)
    assert all(isinstance(v, np.ndarray) for v in splitted_data.values())

def test_split_data_feature_count(cmapss_train_df, cmapss_test_df):
    result = split_data(cmapss_train_df, cmapss_test_df)
    assert result['X_train'].shape[1] == len(FEATURES)
    assert result['X_test'].shape[1] == len(FEATURES)

def test_split_data_xy_consistent(cmapss_train_df, cmapss_test_df):
    result = split_data(cmapss_train_df, cmapss_test_df)
    assert len(result['X_train']) == len(result['y_train'])
    assert len(result['X_test']) == len(result['y_test'])