import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    :param file_path: Path to the file where the object will be saved.
    :param obj: The Python object to be saved.
    """
    try:
        import pickle
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object at {file_path}: {e}")
        raise CustomException(e, sys)
    
def evalute_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # Train the model
            model.fit(X_train, y_train)

            # Predict on test data
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)