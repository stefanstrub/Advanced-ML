import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import VarianceThreshold

x_test = pd.read_csv('X_test.csv')
x_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

x_test.drop('id',inplace=True, axis=1)
x_train.drop('id',inplace=True, axis=1)
y_train.drop('id',inplace=True, axis=1)

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

#### feature detection
p = 100
sel = VarianceThreshold(threshold=100)
x_train_features = sel.fit_transform(x_train_valid)

### regression



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