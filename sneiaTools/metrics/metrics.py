import numpy as np

def mae(predictions, targets):
    targets, predictions = np.array(targets), np.array(predictions)
    mae = np.mean(np.abs(predictions - targets))
    return mae

def mse(predictions, target):
    if len(predictions) != len(target):
        raise ValueError("The length of predictions and target must be the same.")
    squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, target)]
    mse = sum(squared_errors) / len(predictions)
    return mse

def rmse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("The length of predictions and targets must be the same.")
    
    squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
    mse = sum(squared_errors) / len(predictions)
    rmse = np.sqrt(mse)
    return rmse

def re(predictions, targets, return_epsilon = False):
    N = targets.shape[0]
    M = targets.shape[1]
    
    epsilon = np.mean(np.sum(np.abs(targets - predictions) / M, axis=1))
    sigma_e = np.sqrt(np.mean((np.sum(np.abs(targets - predictions) / M, axis = 1) - epsilon) ** 2))

    if return_epsilon:
        return f"{sigma_e:.8f}", f"{epsilon:.8f}"
    else:
        return f"{sigma_e:.8f}"