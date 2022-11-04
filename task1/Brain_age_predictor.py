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

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import r2_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor

file_path = 'task1/'
x_test = pd.read_csv(file_path+'X_test.csv')
x_train = pd.read_csv(file_path+'X_train.csv')
y_train = pd.read_csv(file_path+'y_train.csv')
x_test.drop('id',inplace=True, axis=1)
x_train.drop('id',inplace=True, axis=1)
y_train.drop('id',inplace=True, axis=1)



y_train_np = np.asarray(y_train)

### imputer
# imputer = KNNImputer(n_neighbors=100)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.fit_transform(x_test)


### outlier detection
outliers_fraction = 0.2
clf = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf")
clf.fit(x_train_imputed)
outliers_prediction = clf.predict(x_train_imputed)
outliers =  x_train_imputed[outliers_prediction == -1]
x_train_clean =  x_train_imputed[outliers_prediction == 1]
y_train_clean = y_train_np[outliers_prediction == 1].ravel()

#### feature selection
p = 100
sel = VarianceThreshold(threshold=100)
sel.fit(x_train_clean)
x_train_features = sel.transform(x_train_clean)
x_test_features = sel.transform(x_test_imputed)

# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_iris
# X, y = load_iris(return_X_y=True)
# knn = KNeighborsClassifier(n_neighbors=30)
# sfs = SequentialFeatureSelector(knn, n_features_to_select=30)
# sfs.fit(x_train_features, y_train_clean)

from sklearn.feature_selection import f_regression
X, y = load_iris(return_X_y=True)
f_statistic, p_values = f_regression(x_train_features, y_train_clean)

selector = SelectKBest(f_regression, k=100).fit(x_train_features, y_train_clean)
x_train_features = selector.transform(x_train_features)
x_test_features = selector.transform(x_test_features)
# sfs.get_support()
# x_train_features = sfs.transform(x_train_clean)

# lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(x_train_features, y_train_clean)
# # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# x_train_features = model.transform(x_train_features)

### split data set
train_validation_ratio = 0.999
cut_index = int(len(x_train_features)*train_validation_ratio)
x_validation_features = x_train_features[cut_index:]
y_validation = y_train_clean[cut_index:]
x_train_features = x_train_features[:cut_index]
y_train_clean = y_train_clean[:cut_index]

### normalize x
scaler = StandardScaler() 
scaler.fit(x_train_features)
x_train_features_normalized = scaler.transform(x_train_features)  
# apply same transformation to test data
x_validation_features_normalized = scaler.transform(x_validation_features)  
x_test_features_normalized = scaler.transform(x_test_features)  

# x_train_features_normalized = np.copy(x_train_features)
# x_validation_features_normalized = np.copy(x_validation_features)
# for i in range(len(x_train_features[0])):
#     mean = np.nanmean(x_train_features[:,i])
#     std = np.nanstd(x_train_features[:,i])
#     x_train_features_normalized[:,i] = (x_train_features[:,i]-mean)/std
#     x_validation_features_normalized[:,i] = (x_validation_features[:,i]-mean)/std

### regression
# reg = linear_model.LinearRegression().fit(x_train_features_normalized, y_train_clean)
# y_validation_predicted_linear = reg.predict(x_validation_features_normalized)
# score_linear = reg.score(x_validation_features_normalized, y_validation)
# RMSE_linear = np.sqrt(np.mean((y_validation_predicted_linear-y_validation)**2))
# R2_linear = r2_score(y_validation_predicted_linear,y_validation)

# reg = linear_model.BayesianRidge().fit(x_train_features_normalized, y_train_clean)
# y_validation_predicted_bayes = reg.predict(x_validation_features_normalized)
# score_bayes = reg.score(x_validation_features_normalized, y_validation)
# RMSE_bayes = np.sqrt(np.mean((y_validation_predicted_bayes-y_validation)**2))
# R2_bayes = r2_score(y_validation_predicted_bayes,y_validation)

# reg = linear_model.LogisticRegression().fit(x_train_features_normalized, y_train_clean)
# y_validation_predicted_log = reg.predict(x_validation_features_normalized)
# score = reg.score(x_validation_features_normalized, y_validation)
# RMSE_log = np.sqrt(np.mean((y_validation_predicted_log-y_validation)**2))
# R2_log = r2_score(y_validation_predicted_log,y_validation)

# MLPRegressor
mlp = MLPRegressor(random_state=1, max_iter=3000, hidden_layer_sizes=(100,3)).fit(x_train_features_normalized, y_train_clean)
y_validation_predicted_mlp = mlp.predict(x_validation_features_normalized)
y_train_predicted_mlp = mlp.predict(x_train_features_normalized)
RMSE_mlp = np.sqrt(np.mean((y_validation_predicted_mlp-y_validation)**2))
R2_mlp = r2_score(y_validation_predicted_mlp,y_validation)

print('MLP: score = ', R2_mlp, ' RMSE = ', RMSE_mlp)


# GPR
kernel = RBF(length_scale=[10]*len(x_train_features_normalized.T),length_scale_bounds=[[1,20]]*len(x_train_features_normalized.T))+ WhiteKernel(noise_level=0.2, noise_level_bounds=[0.1,0.5])
gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0, normalize_y=True).fit(x_train_features_normalized, y_train_clean)

y_validation_predicted_gpr = gpr.predict(x_validation_features_normalized)
y_train_predicted_gpr = gpr.predict(x_train_features_normalized)
y_test_predicted_gpr = gpr.predict(x_test_features_normalized)
RMSE_GPR = np.sqrt(np.mean((y_validation_predicted_gpr-y_validation)**2))
R2_GPR = r2_score(y_validation_predicted_gpr,y_validation)

print('GPR: score = ', R2_GPR, ' RMSE = ', RMSE_GPR)

# kernel_mask = gpr.kernel_.length_scale<20
# x_train_features_normalized_reduced = x_train_features_normalized[:,kernel_mask]
# x_validation_features_normalized_reduced = x_validation_features_normalized[:,kernel_mask]

# # GPR
# kernel = RBF(length_scale=[1]*len(x_train_features_normalized_reduced.T),length_scale_bounds=[[0.1,100]]*len(x_train_features_normalized_reduced.T))
# gpr = GaussianProcessRegressor(kernel=kernel,
#         random_state=0, normalize_y=True).fit(x_train_features_normalized_reduced, y_train_clean)

# y_validation_predicted = gpr.predict(x_validation_features_normalized_reduced)
# RMSE_new = np.sqrt(np.mean((y_validation_predicted-y_validation)**2))
# R2_GPR_new = r2_score(y_validation_predicted,y_validation)

fig = plt.figure()
plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation, c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation_predicted_mlp, c='r',
         marker='o', markerfacecolor='None', linestyle='None')
plt.show()

fig = plt.figure()
plt.plot(np.linspace(0, y_train_clean.shape[0] - 1, y_train_clean.shape[0]), y_train_clean, c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_train_clean.shape[0] - 1, y_train_clean.shape[0]), y_train_predicted_mlp, c='g',
         marker='o', markerfacecolor='None', linestyle='None')
plt.show()

fig = plt.figure()
plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation, c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation_predicted_gpr, c='r',
         marker='o', markerfacecolor='None', linestyle='None')
plt.show()

fig = plt.figure()
plt.plot(np.linspace(0, y_train_clean.shape[0] - 1, y_train_clean.shape[0]), y_train_clean, c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_train_clean.shape[0] - 1, y_train_clean.shape[0]), y_train_predicted_gpr, c='g',
         marker='o', markerfacecolor='None', linestyle='None')
plt.show()


prediction = np.zeros((len(x_test),2))
prediction[:,0] = np.arange(len(x_test))
prediction[:,1] = y_test_predicted_gpr 
df = pd.DataFrame(prediction, columns=['id','y'])
df.to_csv('task1/predictions.csv', index=False)

print('end')