import numpy as np

class Metrics:

    def __init__(self):
        pass

    def mae(self, predictions, targets, check_symmetry = True):
        if len(predictions) != len(targets) and check_symmetry:
            raise ValueError("The length of predictions and targets must be the same.")
        targets, predictions = np.array(targets), np.array(predictions)
        mae = np.mean(np.abs(predictions - targets))
        return mae

    def mse(self, predictions, target, check_symmetry = True):
        if len(predictions) != len(target) and check_symmetry:
            raise ValueError("The length of predictions and target must be the same.")
        squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, target)]
        mse = sum(squared_errors) / len(predictions)
        return mse

    def rmse(self, predictions, targets, check_symmetry = True):
        if len(predictions) != len(targets) and check_symmetry:
            raise ValueError("The length of predictions and targets must be the same.")
        
        squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
        mse = sum(squared_errors) / len(predictions)
        rmse = np.sqrt(mse)
        return rmse

    def re(self, predictions, targets, return_epsilon = False, check_symmetry = True):
        if len(predictions) != len(targets) and check_symmetry:
            raise ValueError("The length of predictions and targets must be the same.")
        
        N = targets.shape[0]
        M = targets.shape[1]
        
        epsilon = np.mean(np.sum(np.abs(targets - predictions) / M, axis=1))
        sigma_e = np.sqrt(np.mean((np.sum(np.abs(targets - predictions) / M, axis = 1) - epsilon) ** 2))

        if return_epsilon:
            return f"{sigma_e:.8f}", f"{epsilon:.8f}"
        else:
            return f"{sigma_e:.8f}"