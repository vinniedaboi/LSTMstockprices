import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
scaler = MinMaxScaler(feature_range=(0, 1))
df = pd.read_csv('VERIFY.csv')
df1=df.reset_index()['High']
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
df1=scaler.inverse_transform(df1).tolist()
plt.plot(df1)
plt.show()