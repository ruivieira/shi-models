import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import click
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import joblib

logger = logging.getLogger(__name__)


def create_offset(dataset, history=1):
    x, y = [], []
    for i in range(len(dataset) - history - 1):
        a = dataset[i:(i + history), 0]
        x.append(a)
        y.append(dataset[i + history, 0])
    return np.array(x), np.array(y)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Trains a model from processed data into a joblib-serialised model"""

    # load processed dataset
    logger.info("Loading raw dataset")
    _df: pd.DataFrame = pd.read_csv(os.path.join(input_filepath, "data.csv"))

    Y = np.array(_df["y"]).reshape(-1, 1)

    N = len(Y)
    print(N)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(Y)

    train_size = 250
    train, test = data[:train_size, :], data[train_size:N, :]

    history = 1
    X_train, y_train = create_offset(train, history)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, history)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=125, batch_size=2, verbose=2)
    tf.keras.models.save_model(model,
                               os.path.join(output_filepath, "model.h5"),
                               save_format="h5",
                               overwrite=True,
                               include_optimizer=True)

    # save scaler
    joblib.dump(scaler, os.path.join(output_filepath, "scaler.joblib"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
