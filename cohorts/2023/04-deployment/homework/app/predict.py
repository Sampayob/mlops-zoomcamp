"""prediction script"""

import argparse
import pickle

import pandas as pd
from flask import Flask, jsonify, request


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def prepare_features(dates):
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{str(dates['year'])}-{str(dates['month'])}.parquet"
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[['PULocationID', 'DOLocationID'] ]= df[['PULocationID', 'DOLocationID']].fillna(-1).astype('int').astype('str')
    return df[['PULocationID', 'DOLocationID']]


def predict(features):
    dv, model = load_model()
    dicts = features.to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return float(f"{y_pred.mean():.2f}")


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    dates = request.get_json()
    features = prepare_features(dates)
    pred = predict(features)
    
    result = {
        'Mean predicted duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
