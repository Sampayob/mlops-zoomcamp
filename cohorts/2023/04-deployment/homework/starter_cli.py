import argparse
import pickle

import pandas as pd


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def prepare_features(config):
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{str(config['year'])}-{str(config['month'])}.parquet"
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].fillna(-1).astype('int').astype('str')
    return df[['PULocationID', 'DOLocationID']]


def predict(config):
    dv, model = load_model()
    features = prepare_features(config)
    dicts = features.to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(f"{y_pred.mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yellow-tripdata-model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-y", "--year", default="2022", help="data year")
    parser.add_argument("-m", "--month", default="01", help="data month")
    args = parser.parse_args()
    config = vars(args)
    predict(config)
