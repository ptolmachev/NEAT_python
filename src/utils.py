import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from pathlib import Path
def get_project_root():
    return Path(__file__).parent.parent

def jsonify(dct):
    dct_jsonified = {}
    for key in dct.keys():
        if type(dct[key]) == type({}):
            dct_jsonified[key] = jsonify(dct[key])
        elif type(dct[key]) == np.ndarray:
            dct_jsonified[key] = dct[key].tolist()
        elif type(dct[key]) == np.int64:
            dct_jsonified[key] = int(dct[key])
        else:
            dct_jsonified[key] = dct[key]
    return dct_jsonified


if __name__ == '__main__':
    print(get_project_root())
