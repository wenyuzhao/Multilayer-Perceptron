import numpy as np
import random

def binary(array, count=1):
    array = np.copy(array)
    for _ in range(count):
        i = random.randrange(array.shape[0])
        array[i] = 1. if array[i] != 1. else 0.
    return array