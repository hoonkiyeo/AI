import csv
import pandas as pd
import numpy as np


def load_data(filepath):
    lst = []
    with open(filepath) as file:
        reader = csv.DictReader(file)
        for row in reader:
            lst.append(row)
    return lst

def calc_features(row):
    x1 = int(row['Attack'])
    x2 = int(row['Sp. Atk'])
    x3 = int(row['Speed'])
    x4 = int(row['Defense'])
    x5 = int(row['Sp. Def'])
    x6 = int(row['HP'])
    return np.array((x1,x2,x3,x4,x5,x6)).reshape(6,)