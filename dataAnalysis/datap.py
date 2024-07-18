import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

my_reader = get_reader('./sample.pcap')
csi_data = my_reader.read_file('./sample.pcap')
csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric='amplitude')

csi_matrix_first = csi_matrix [:, :, 0, 0]
csi_matrix_squeezd = np.squeeze(csi_matrix_first)
finite_mask = np.isfinite(csi_matrix_squeezd).all(axis=1)
csi_matrix_squeezd = csi_matrix_squeezd[finite_mask]

for x in range(len(csi_matrix_squeezd)):
    csi_matrix_squeezd[x] = lowpass(csi_matrix_squeezd[x], 10, 100, 5)
    csi_matrix_squeezd[x] = hampel(csi_matrix_squeezd[x], 10, 3)
    csi_matrix_squeezd[x] = running_mean(csi_matrix_squeezd[x], 10)

null_subcarriers = [-64, -63, -62, -61, -60, -59, -1, 0, 1, 59, 60, 61, 62, 63]
pilot_subcarriers = [11, 25, 53, -11, -25, -53]
removed_subcarriers = null_subcarriers + pilot_subcarriers
removed_subcarriers.sort(reverse=True)

for i in removed_subcarriers:
    csi_matrix_squeezd = np.delete(csi_matrix_squeezd, i + 64, 1)

# get the average value of the subcarriers
df_csi = pd.DataFrame(csi_matrix_squeezd.mean(axis=1), columns=['csi'])
columns = ['time', 'temp', 'rh']
df_temp = pd.read_csv('./sample_temps.csv', header=None, names=columns)

print(len(df_temp))
interval_size = len(df_csi) / len(df_temp)
print(interval_size)


df_csi['group'] = df_csi.index // interval_size
df_avg_csi = df_csi.groupby('group').mean().reset_index()
if (len(df_avg_csi) > len(df_temp)):
    df_avg_csi = df_avg_csi.head(len(df_temp))
else:
    df_temp = df_temp.head(len(df_avg_csi))

df_combined = pd.concat([df_temp.reset_index(drop=True), df_avg_csi], axis=1)

# # # now the machine learning
x_rf = df_combined[['temp']]
y_rf = df_combined['csi']

model = RandomForestRegressor()
# model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x_rf, y_rf, test_size=0.5, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# kfold training
k = 5
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Perform K-Fold Cross-Validation
mse_list = []

for train_index, test_index in kf.split(df_combined):
    # Split data
    X_train, X_test = df_combined[['temp']].iloc[train_index], df_combined[['temp']].iloc[test_index]
    y_train, y_test = df_combined['csi'].iloc[train_index], df_combined['csi'].iloc[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# machine learning graphing
plt.scatter(df_combined['temp'], df_combined['csi'], color='blue')
y_pred = model.predict(x_test)
plt.scatter(x_test, y_pred, color='red')
plt.legend()
plt.show()

# polynomial
# model = np.poly1d(np.polyfit(df_combined['csi'], df_combined['temp'], 3))
# polyline = np.linspace(min(df_combined['csi']), max(df_combined['csi']), 1000)
# print(polyline)
# plt.scatter(df_combined['csi'], df_combined['temp'])
# plt.plot(polyline, model(polyline))
# plt.show()