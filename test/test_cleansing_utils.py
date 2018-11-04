from main.cleansing_utils import CleansingUtils as Cu
from logging import DEBUG
import pandas as pd

orig_data = pd.read_csv('../data/test.csv')
col_names = orig_data.columns
cu = Cu()
cu.log_level = DEBUG


def test___init__():
    test_obj = Cu()
    test_obj.log_level = DEBUG


def test_fill_nan_mean_int():
    col = 'int'
    test_data = orig_data.copy()
    test_data = cu.fill_nan_mean(test_data, col, 'int')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_mean_float():
    test_data = orig_data.copy()
    col = 'float'
    test_data = cu.fill_nan_mean(test_data, col, 'float')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_median_int():
    test_data = orig_data.copy()
    col = 'int'
    test_data = cu.fill_nan_median(test_data, col, 'int')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_median_float():
    test_data = orig_data.copy()
    col = 'float'
    test_data = cu.fill_nan_median(test_data, col, 'float')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_mode_int():
    test_data = orig_data.copy()
    col = 'int'
    test_data = cu.fill_nan_mode(test_data, col, 'int')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_mode_float():
    test_data = orig_data.copy()
    col = 'float'
    test_data = cu.fill_nan_mode(test_data, col, 'float')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_range_int():
    test_data = orig_data.copy()
    col = 'int'
    test_data = cu.fill_nan_range(test_data, col, cast_type='int')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_range_float():
    test_data = orig_data.copy()
    col = 'float'
    test_data = cu.fill_nan_range(test_data, col, cast_type='float')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_user_range_int():
    test_data = orig_data.copy()
    col = 'int'
    test_data = cu.fill_nan_user_range(test_data, col,
                                       data_max=100, data_min=-150,
                                       cast_type='int')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_user_range_float():
    test_data = orig_data.copy()
    col = 'float'
    test_data = cu.fill_nan_user_range(test_data, col,
                                       data_max=100.52, data_min=-150.2,
                                       cast_type='float')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_range_date_yyyymmdd():
    test_data = orig_data.copy()
    col = 'Date'
    test_data = cu.fill_nan_range_date(test_data, col, date_fmt='%Y/%m/%d')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_range_date_timestamp():
    test_data = orig_data.copy()
    col = 'Datetime'
    test_data = cu.fill_nan_range_date(test_data, col, date_fmt='%Y%m%d %H:%M:%S')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_user_range_date():
    test_data = orig_data.copy()
    col = 'Datetime'
    data_max = pd.to_datetime('2020-12-31 23:59:59')
    data_min = pd.to_datetime('2010-01-01 00:00:00')
    test_data = cu.fill_nan_user_range_date(test_data, col,
                                            data_max=data_max, data_min=data_min,
                                            date_fmt='%Y%m%d %H:%M:%S')
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_list():
    test_data = orig_data.copy()
    col = 'ID'
    test_list = ['0001', '0002', '0003']
    test_data = cu.fill_nan_list(test_data, col, test_list)
    assert test_data[col].isnull().sum() == 0


def test_fill_nan_list_weights():
    test_data = orig_data.copy()
    col = 'ID'
    test_list = ['0001', '0002', '0003']
    weights = [1, 0, 0]
    test_data = cu.fill_nan_list(test_data, col, test_list, weights)
    assert test_data[col].isnull().sum() == 0


def test_create_current_weights_names():
    test_data = orig_data.copy()
    col = 'str'
    test_weights, test_names = cu.create_current_weights_names(test_data, col)
    assert test_weights == [1/5, 1/5, 1/5, 1/5, 1/5]
    assert set(test_names) == set(['IDNull0001', 'DateNull0002',
                                   'DatetimeNull0003', 'intNull0004',
                                   'floatNull0005'])


def test_update_dataframe():
    test_data_target = pd.DataFrame({'A': [-20, -10, 0, 10, 20],
                                     'B': [1, 2, 3, 4, 5],
                                     'C': ['a', 'b', 'b', 'b', 'a']})
    test_data_source = pd.DataFrame({'A': [-20, -10, 0, 10, 20],
                                     'B': [1, 2, 3, 4, 5],
                                     'C': ['A', 'B', 'B', 'B', 'A']})
    test_data = cu.update_dataframe(test_data_target, test_data_source, 'B', 'B')
    assert (test_data_source == test_data).all().all()


def test_missing_rate():
    test_data = orig_data.copy()
    missing_rate, total_missing = cu.missing_rate(test_data)
    assert total_missing == 1/6
