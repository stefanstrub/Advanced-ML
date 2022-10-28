import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import VarianceThreshold

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

file_path = 'task1/'
x_test = pd.read_csv(file_path+'X_test.csv')
x_train = pd.read_csv(file_path+'X_train.csv')
y_train = pd.read_csv(file_path+'y_train.csv')
x_test.drop('id',inplace=True, axis=1)
x_train.drop('id',inplace=True, axis=1)
y_train.drop('id',inplace=True, axis=1)



y_train_np = np.asarray(y_train)

### imputer
imputer = KNNImputer(n_neighbors=4)
x_train_imputed = imputer.fit_transform(x_train)

### outlier detection
outliers_fraction = 0.05
clf = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf")
clf.fit(x_train_imputed)
outliers_prediction = clf.predict(x_train_imputed)
outliers =  x_train_imputed[outliers_prediction == -1]
x_train_valid =  x_train_imputed[outliers_prediction == 1]
y_train_valid = y_train_np[outliers_prediction == 1]

#### feature detection
p = 100
sel = VarianceThreshold(threshold=1000)
x_train_features = sel.fit_transform(x_train_valid)

train_validation_ratio = 0.7
cut_index = int(len(x_train_features)*train_validation_ratio)
x_validation_features = x_train_features[cut_index:]
y_validation = y_train_valid[cut_index:]
x_train_features = x_train_features[:cut_index]
y_train_valid = y_train_valid[:cut_index]

### normalize x
x_train_features_normalized = np.copy(x_train_features)
for i in range(len(x_train_features[0])):
    mean = np.nanmean(x_train_features[:,i])
    std = np.nanstd(x_train_features[:,i])
    x_train_features_normalized[:,i] = (x_train_features[:,i]-mean)/std

x_validation_features_normalized = np.copy(x_validation_features)
for i in range(len(x_validation_features[0])):
    mean = np.nanmean(x_validation_features[:,i])
    std = np.nanstd(x_validation_features[:,i])
    x_validation_features_normalized[:,i] = (x_validation_features[:,i]-mean)/std

### regression
reg = linear_model.LinearRegression().fit(x_train_features_normalized, y_train_valid)
y_validation_predicted = reg.predict(x_validation_features_normalized)
score_linear = reg.score(x_validation_features_normalized, y_validation)
RMSE_linear = np.sqrt(np.mean((y_validation_predicted-y_validation)**2))

reg = linear_model.BayesianRidge().fit(x_train_features_normalized, y_train_valid)
y_validation_predicted = reg.predict(x_validation_features_normalized)
score_bayes = reg.score(x_validation_features_normalized, y_validation)
RMSE_bayes = np.sqrt(np.mean((y_validation_predicted-y_validation)**2))

reg = linear_model.LogisticRegression().fit(x_train_features_normalized, y_train_valid)
y_validation_predicted = reg.predict(x_validation_features_normalized)
score = reg.score(x_validation_features_normalized, y_validation)
RMSE = np.sqrt(np.mean((y_validation_predicted-y_validation)**2))

# GPR
kernel = RBF(length_scale=[1]*len(x_train_features_normalized.T),length_scale_bounds=[[0.1,10]]*len(x_train_features_normalized.T))
gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0, normalize_y=True).fit(x_train_features_normalized, y_train_valid)

score = gpr.score(x_train_features_normalized, y_train_valid)

y_validation_predicted = gpr.predict(x_validation_features_normalized)
score = gpr.score(x_validation_features_normalized, y_validation)
RMSE = np.sqrt(np.mean((y_validation_predicted-y_validation)**2))


x_train_np = np.asarray(x_train_valid)
x_train_np_norm = np.copy(x_train_np)
for i in range(len(x_train_np[0])):
    mean = np.nanmean(x_train_np[:,i])
    std = np.nanstd(x_train_np[:,i])
    x_train_np_norm[:,i] = (x_train_np[:,i]-mean)/std

indecies = np.arange(len(x_train_np[0]))
figure = plt.figure()
for i in range(len(x_train_imputed)-1200) :
    plt.plot(indecies,x_train_imputed[i,:])
plt.show()

indecies = np.arange(len(x_train_np_norm[0]))
figure = plt.figure()
for i in range(len(x_train_np_norm)-1200) :
    plt.plot(indecies,x_train_np_norm[i,:])
plt.show()

print('end')