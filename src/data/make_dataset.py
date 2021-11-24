# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pssm.structure import UnivariateStructure
from pssm.dglm import NormalDLM
import numpy as np
from typing import List
import pandas as pd
import os


def generate_anomalous_data(structure: UnivariateStructure,
                            n_obs: int,
                            m0: np.ndarray,
                            c0: np.ndarray,
                            v: float,
                            anomaly_times: List[int],
                            ratio: float) -> List[float]:
    ndlm = NormalDLM(structure=structure, V=v)
    state0 = np.random.multivariate_normal(m0, c0)

    states = [state0]

    for t in range(1, n_obs):
        states.append(ndlm.state(states[t - 1]))

    for t in anomaly_times:
        states[t] = states[t] * ratio

    obs = [None]
    for t in range(1, n_obs):
        obs.append(ndlm.observation(states[t]))

    return obs[1:]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # Generate the raw dataset
    logger.info('simulating raw data')

    np.random.seed(23)
    period = 7

    structure = UnivariateStructure.locally_constant(1.4) + \
                UnivariateStructure.cyclic_fourier(period=period,
                                                   harmonics=1,
                                                   W=np.identity(2) * 2)
    m0 = np.array([100, 0, 0])
    c0 = np.identity(3)

    raw = generate_anomalous_data(structure=structure,
                                  n_obs=365,
                                  m0=m0,
                                  c0=c0,
                                  v=2.5,
                                  anomaly_times=[27, 53, 270],
                                  ratio=3.0)

    raw_df = pd.DataFrame(raw, columns=["y"])
    raw_df['day'] = list(range(364))
    raw_df.to_csv(os.path.join(input_filepath, "data.csv"),
                  index_label="t")

    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
