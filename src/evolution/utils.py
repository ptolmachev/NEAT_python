import numpy as np

def standardize(vals):
    if len(vals) == 0:
        return np.array([0.5])
    else:
        if np.min(vals) != np.max(vals):
            return (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        else:
            return 0.5 * np.ones_like(vals)

def sigmoid(x, mu, slope):
    return 1.0 / (1 + np.exp(-slope * (x - mu)))

def get_probs(values, slope):
    values_standardized = standardize(values)
    tmp = np.exp(slope * values_standardized)
    probs = tmp/np.sum(tmp)
    return probs

def jsonify(dct):
    dct_jsonified = {}
    for key in list(dct.keys()):
        if type(dct[key]) == type({}):
            dct_jsonified[key] = jsonify(dct[key])
        elif type(dct[key]) == np.ndarray:
            dct_jsonified[key] = dct[key].tolist()
        else:
            dct_jsonified[key] = dct[key]
    return dct_jsonified