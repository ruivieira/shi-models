from typing import Iterable, Dict, Union, List
import tensorflow as tf
import numpy as np
import os
import joblib

class SHIModel(object):
    result = {}

    def __init__(self):
        self.loaded = False
        self.model = None
        self.scaler = None

    def load(self):
        print("Loading model", os.getpid())
        self.model = tf.keras.models.load_model('./models/model.h5', custom_objects=None, compile=False)
        self.scaler = joblib.load("./models/scaler.joblib")
        self.loaded = True
        print("Loaded model")

    def predict(self,
                X: np.ndarray,
                names: Iterable[str],
                meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        if not self.loaded:
            self.load()
        y = X.reshape(-1, 1)
        prediction = self.model.predict(self.scaler.transform(y))
        return self.scaler.inverse_transform(prediction)

    def tags(self):
        return self.result