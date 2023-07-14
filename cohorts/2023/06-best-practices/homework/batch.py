#!/usr/bin/env python
# coding: utf-8

import logging
import pickle
import sys
from pathlib import Path
from typing import List
import os

import pandas as pd
from pandas import DataFrame

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


options = {
    'client_kwargs': {
        'endpoint_url':os.getenv('S3_ENDPOINT_URL')
    }
}


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(input_path, year, month) -> DataFrame:
    """Read raw data."""

    if os.environ.get('S3_ENDPOINT_URL'):
        options = {
        'client_kwargs': {
            'endpoint_url': os.getenv('S3_ENDPOINT_URL')}
        }
        return pd.read_parquet(f"s3://{os.getenv('S3_BUCKET_NAME')}/in/{year:04d}-{month:02d}.parquet", storage_options=options)

    return pd.read_parquet(input_path)


def prepare_data(df: DataFrame, categorical: List, month: str, year: str) -> DataFrame:
    """Transform raw data into processed data."""
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    return df


def predict(df, categorical_cols, model, dv):
    """Return model prediction."""

    dicts = df[categorical_cols].to_dict(orient="records")
    X_val = dv.transform(dicts)
    prediction = model.predict(X_val)
    return prediction


def save_data(df, y_pred, output_path):
    """Save output DataFrame."""
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred
   
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_result.to_parquet(output_path, engine="pyarrow", index=False)


def main(year, month):
    """Main function."""
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ["PULocationID", "DOLocationID"]

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    logging.info("Read data: yellow_tripdata_%s-%s.parquet", year, month)
    df = read_data(input_file, year ,month)

    logging.info("Prepare data")
    df = prepare_data(df, categorical, month, year)

    logging.info("Train data")
    y_pred = predict(df, categorical, lr, dv)

    logging.info("Predicted mean duration: %s", round(y_pred.mean(), 2))
    logging.info("Predicted sum duration: %s", round(sum(y_pred), 2))

    logging.info("Save data: %s", output_file)
    save_data(df, y_pred, output_file)
 

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
