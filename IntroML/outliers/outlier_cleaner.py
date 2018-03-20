#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):  # y_pred, x, y
    """
        Clean away the 10% of points that have the largest residual errors
        (difference between the prediction and the actual net worth).

        Return a list of tuples named cleaned_data where each tuple is of the form (age, net_worth, error).
    """
    size = len(predictions)
    rm_size = int(size*0.1)

    diff = abs(predictions - net_worths)    # absolute errors

    print(len(diff))
    for i in range(rm_size):
        maxvalue = max(diff)
        idx = np.where(diff == maxvalue)

        diff = np.delete(diff, idx)
        ages = np.delete(ages, idx)
        net_worths = np.delete(net_worths, idx)

    print(len(diff))

    cleaned_data = (ages, net_worths)

    return cleaned_data
