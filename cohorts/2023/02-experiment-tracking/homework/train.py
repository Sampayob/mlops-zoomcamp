import os
import pickle
import click

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment-train")
mlflow.sklearn.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        max_depth=10
        mlflow.log_param("max_depth", 10)
        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()

# Q1: 2.3.2
# Q2: 154KB
# Q3: 10
# Q4: 2.45
# Q5: 2.285