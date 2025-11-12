import os
import sys

import numpy as np
import pandas as pd
import dill
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