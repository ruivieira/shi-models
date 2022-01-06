import logging
from typing import Iterable, Dict, Union, List
import numpy as np
import os
import joblib
import datetime
import json
import uuid

import requests
from cloudevents.http import CloudEvent, to_json, to_structured

logger = logging.getLogger('SHIModel')
OB_CLIENT_URI = os.getenv('OB_CLIENT_URI')


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

        current_load = request.get("current load")
        estimated_load = np.asscalar(predicted_load)
        e = current_load - estimated_load

        _id = str(uuid.uuid4())

        # Create POST CloudEvent object
        post_attributes = {
            "id": _id,
            "type": "org.drools.model.HostLoad",
            "source": "example",
            "datacontenttype": "application/json",
            "obclienturi": OB_CLIENT_URI,
        }
        post_data = {
            "host": request.get("host"),
            "current load": request.get("current load")
        }
        post_response = CloudEvent(post_attributes, post_data)
        post_headers, post_body = to_structured(post_response)
        print(json.loads(to_json(post_response)))
        try:
            requests.post(OB_CLIENT_URI, data=post_body, headers=post_headers)
        except requests.exceptions.RequestException as ex:
            logger.error("Error sending CloudEvent to %s", OB_CLIENT_URI)
            logger.error(ex)

        # Create this endpoints response CloudEvent object
        response_data = {
            "id": _id,
            "host": request.get("host"),
            "current load": current_load,
            "estimated load": estimated_load,
            "when": request.get("when"),
            "e": e,
            "diagnosis": self._diagnosis(e)
        }
        response_event = CloudEvent(post_attributes, response_data)
        print(json.loads(to_json(response_event)))
        return json.loads(to_json(response_event))

    def tags(self):
        return self.result
