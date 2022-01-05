from typing import Iterable, Dict, Union, List
import numpy as np
import os
import joblib
import datetime


class SHIModel(object):
    result = {}

    def __init__(self):
        self.loaded = False
        self.model = None
        self.sigma = None

    def load(self):
        print("Loading model", os.getpid())
        self.model = joblib.load("./models/model.joblib")
        self.sigma = joblib.load("./models/sigma.joblib")
        self.loaded = True
        print("Loaded model")

    def predict(self,
                X: np.ndarray,
                names: Iterable[str],
                meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        if not self.loaded:
            self.load()
        y = X.reshape(-1, 1)
        prediction = np.array(self.model.predict(y)).reshape(-1, 1)
        return prediction

    def _diagnosis(self, e):
        if e > 3 * self.sigma:
            return "dangerously high load"
        elif e > 2 * self.sigma:
            return "very high load"
        elif e > self.sigma:
            return "high load"
        elif abs(e) < self.sigma:
            return "normal load"
        else:
            return "low load"

    def predict_raw(self, request):
        if not self.loaded:
            self.load()
        print(request)
        # get day number for ISO 8601
        day = datetime.datetime.strptime(request.get("when"), "%Y-%m-%dT%H:%M:%S.%f%z").timetuple().tm_yday
        day = np.array(day).reshape(-1, 1)
        predicted_load = self.model.predict(day)

        response = {"estimated load": np.asscalar(predicted_load),
                    "current load": request.get("current load"),
                    "when": request.get("when"),
                    "host": request.get("host")}
        e = response["current load"] - response["estimated load"]

        response["e"] = e
        response["diagnosis"] = self._diagnosis(e)

        return response

    def tags(self):
        return self.result
