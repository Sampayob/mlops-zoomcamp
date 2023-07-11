#!/usr/bin/env python
# coding: utf-8

import logging
import pickle
import sys
from pathlib import Path
from typing import List

import pandas as pd
from pandas import DataFrame

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def read_data(filename: str) -> DataFrame:
    """Read raw data."""
    return pd.read_parquet(filename)


def prepare_data(df: DataFrame, categorical: List, month: str, year: str) -> DataFrame:
    """Transform raw data into processed data."""
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    return df


def main(year, month):
    """Main function."""
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/yellow_tripdata_{year:04d}-{month:02d}.parquet"

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]

    logging.info("Read data: yellow_tripdata_%s-%s.parquet", year, month)
    df = read_data(input_file)

    logging.info("Prepare data")
    df = prepare_data(df, categorical, month, year)

    logging.info("Training data")
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    logging.info("Predicted mean duration: %s", y_pred.mean())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    logging.info("Save data: %s", output_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_result.to_parquet(output_file, engine="pyarrow", index=False)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
