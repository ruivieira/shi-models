import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import click
import pandas as pd
import os
import joblib
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Trains a model from processed data into a joblib-serialised model"""

    # load processed dataset
    logger.info("Loading raw dataset")
    _df: pd.DataFrame = pd.read_csv(os.path.join(input_filepath, "data.csv"))

    # remove anomalies from training dataset
    Y = _df['y']
    Y_mean = Y.mean()
    Y_std = Y.std()

    bounds = [Y_mean + 3 * Y_std, Y_mean - 3 * Y_std]
    _df['y_train'] = _df['y'].apply(
        lambda x: x if bounds[1] < x < bounds[0] else Y_mean)

    model = XGBRegressor()
    model.fit(_df.day, _df.y_train)

    # save model
    joblib.dump(model, os.path.join(output_filepath, "model.joblib"))
    joblib.dump(Y_std, os.path.join(output_filepath, "sigma.joblib"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
