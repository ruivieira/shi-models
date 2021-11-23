shi-models
==============================

SHI test models.

## Usage

- `make data` simulates the dataset into `data/raw/data.csv`
- `make train` created the serialised model and scaler (respectively in `models/model.h5` and `models/scaler.joblib`)
- `make s2i` creates a container image (Seldon) named `ruvieira/shi-model:latest`

You only need to call one target (_ie_ `make s2i` will call the `data` and `train` targets as dependencies).
## Service

To create the Seldon model micro-service run

```shell
$ make s2i
```

This will create a container image called `ruivieira/shi-model:latest` which you can run locally with:

```shell
$ docker run -p 6000:6000 -p 9000:9000 ruivieira/shi-model
```

The prediction REST endpoint is `GET http://localhost:9000/predict` and the payload
has the format

```json
{
  "data": {
    "ndarray": [[ $VALUE ]]
  }
}
```

As an example:

```shell
$ curl -g http://localhost:9000/predict --data-urlencode 'json={"data":{"ndarray":[[109.0]]}}'
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
