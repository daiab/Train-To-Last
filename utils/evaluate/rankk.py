"""
gallery
probe
base
pair method
"""
import numpy as np

def read_id_feature(file_name):
    data = np.genfromtxt(file_name, dtype=float, delimiter=' ')
    id, feature = np.split(data, [1], axis=1)
    return id.astype(int), feature

