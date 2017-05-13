import numpy as np
import random
from . import noise

X = [
    np.array([
        0,0,1,0,0,
        0,1,1,1,0,
        1,0,1,0,1,
        0,0,1,0,0,
        0,0,1,0,0,
    ], dtype=float),
    np.array([
        0,0,1,0,0,
        0,0,0,1,0,
        1,1,1,1,1,
        0,0,0,1,0,
        0,0,1,0,0,
    ], dtype=float),
    np.array([
        0,0,1,0,0,
        0,0,1,0,0,
        1,0,1,0,1,
        0,1,1,1,0,
        0,0,1,0,0,
    ], dtype=float),
    np.array([
        0,0,1,0,0,
        0,1,0,0,0,
        1,1,1,1,1,
        0,1,0,0,0,
        0,0,1,0,0,
    ], dtype=float),
]

# Add noise data
X.append(noise.binary(X[0]))
X.append(noise.binary(X[1]))
X.append(noise.binary(X[2]))
X.append(noise.binary(X[3]))

Y = [
    np.array([1,0,0,0], dtype=float),
    np.array([0,1,0,0], dtype=float),
    np.array([0,0,1,0], dtype=float),
    np.array([0,0,0,1], dtype=float),
    np.array([1,0,0,0], dtype=float),
    np.array([0,1,0,0], dtype=float),
    np.array([0,0,1,0], dtype=float),
    np.array([0,0,0,1], dtype=float),
]

train_data = [
    (X[0], Y[0]),
    (X[1], Y[1]),
    (X[2], Y[2]),
    (X[3], Y[3]),
]

test_data = [
    (noise.binary(X[0]), Y[0]),
    (noise.binary(X[1]), Y[1]),
    (noise.binary(X[2]), Y[2]),
    (noise.binary(X[3]), Y[3]),
]

def display(x):
    x = x.reshape(25)
    def strof(i):
        if x[i] > 0.75:   return '■'
        elif x[i] > 0.5:  return '▥'
        elif x[i] > 0.25: return '□'
        else:             return ' '
    for i in range(0, 25, 5):
        print("{} {} {} {} {}".format(strof(i), strof(i+1), strof(i+2), strof(i+3), strof(i+4)))

