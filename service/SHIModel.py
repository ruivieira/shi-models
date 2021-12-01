from typing import Iterable, Dict, Union, List
import numpy as np
import os
import joblib

class SHIModel(object):
    result = {}

    def __init__(self):
        self.loaded = False
        self.model = None

    def load(self):
        print("Loading model", os.getpid())
        self.model = joblib.load("./models/model.joblib")
        self.loaded = True
        print("Loaded model")

    def predict(self,
                X: np.ndarray,
                names: Iterable[str],
                meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        if not self.loaded:
            self.load()
        y = X.reshape(-1, 1)
        prediction = self.model.predict(y)
        return prediction

    def tags(self):
        return self.result