import os
from unittest.mock import Mock
from datetime import datetime

import numpy as np
from pandas import DataFrame
import pandas as pd

from batch import (
    prepare_data,
    read_data,
    get_input_path,
    get_output_path,
    predict,
    save_data)


def test_get_input_data():
    year = 2022
    month = 2
    expected_input_path = get_input_path(year, month)
    assert expected_input_path == os.getenv('INPUT_FILE_PATTERN').format(year=year, month=month), \
        "expected_input_path is not correct."
    
def test_get_output_data():
    year = 2022
    month = 2
    expected_input_path = get_output_path(year, month)
    assert expected_input_path == os.getenv('OUTPUT_FILE_PATTERN').format(year=year, month=month), \
        "expected_input_path is not correct."


def test_read_data():
    year = 2022
    month = 2
    input_path = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet'
    expected_data = read_data(input_path, year, month)

    assert isinstance(expected_data, DataFrame), "The expected data is not a Pandas DataFrame."


def test_prepare_data(df_input):
    expected_data = [
        (
            "-1",
            "-1",
            datetime.strptime("2022-01-01 01:02:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2022-01-01 01:10:00", "%Y-%m-%d %H:%M:%S"),
            8.0,
            "2022/01_0",
        ),
        (
            "1",
            "-1",
            datetime.strptime("2022-01-01 01:02:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2022-01-01 01:10:00", "%Y-%m-%d %H:%M:%S"),
            8.0,
            "2022/01_1",
        ),
        (
            "1",
            "2",
            datetime.strptime("2022-01-01 02:02:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2022-01-01 02:03:00", "%Y-%m-%d %H:%M:%S"),
            1.0,
            "2022/01_2",
        ),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "duration",
        "ride_id",
    ]

    expected_data = pd.DataFrame(expected_data, columns=columns)
    categorical = ["PULocationID", "DOLocationID"]
    year = 2022
    month = 1

    assert expected_data.equals(prepare_data(df_input, categorical, month, year))


def test_predict(df_input):
    """Return model prediction."""
    categorical_cols = ["PULocationID", "DOLocationID"]

    model = Mock()
    dv = Mock()
    
    predict(df_input, categorical_cols, model, dv)

    dv.transform.assert_called_once()
    model.predict.assert_called_once()


def test_save_data(df_input):
    """Save output DataFrame."""
    y_pred = np.ones(len(df_input))
    df_input['ride_id'] = np.ones(len(df_input))
    output_path='test_save_data.parquet'
    save_data(df_input, y_pred, output_path)
    assert 'test_save_data.parquet' in os.listdir()
    os.remove(output_path)
    assert 'test_save_data.parquet' not in os.listdir()
