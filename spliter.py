import pandas as pd
import numpy as np

df = pd.read_csv('heart.csv')

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
test = df[~msk]

train.to_csv('heart_train.csv')
test.to_csv('heart_test.csv')