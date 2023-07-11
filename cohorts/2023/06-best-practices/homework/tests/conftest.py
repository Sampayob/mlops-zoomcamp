from datetime import datetime

import pandas as pd
import pytest
from pandas import DataFrame


@pytest.fixture(scope="module")
def input_data() -> DataFrame:
    """ """

    def dt(hour, minute, second=0):
        return datetime(2022, 1, 1, hour, minute, second)

    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture(scope="module")
def expected_data() -> DataFrame:
    """ """
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

    return pd.DataFrame(expected_data, columns=columns)
