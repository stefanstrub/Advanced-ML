import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer, SimpleImputer

from sklearn.metrics import r2_score

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel

# file_path = 'task1/'
file_path = ''
x_test = pd.read_csv(file_path + 'X_test.csv')
x_train = pd.read_csv(file_path + 'X_train.csv')
y_train = pd.read_csv(file_path + 'y_train.csv')
x_test.drop('id', inplace=True, axis=1)
x_train.drop('id', inplace=True, axis=1)
y_train.drop('id', inplace=True, axis=1)

y_train_np = np.asarray(y_train)

### imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)

### outlier detection
outliers_fraction = 0.1
clf = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf")
clf.fit(x_train_imputed)
outliers_prediction = clf.predict(x_train_imputed)
outliers = x_train_imputed[outliers_prediction == -1]
x_train_clean = x_train_imputed[outliers_prediction == 1]
y_train_clean = y_train_np[outliers_prediction == 1]

#### feature detection
p = 100
sel = VarianceThreshold(threshold=1000)
x_train_features = sel.fit_transform(x_train_clean)

selector = SelectKBest(f_regression, k=90)
selector.fit(x_train_features, y_train_clean[:, 0])
scores = -np.log10(selector.pvalues_)
scores /= scores.max()

X_indices = np.arange(x_train_features.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(X_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()

x_train_features = x_train_features[:, selector.get_support()]

train_validation_ratio = 0.8
cut_index = int(len(x_train_features) * train_validation_ratio)
x_validation_features = x_train_features[cut_index:]
y_validation = y_train_clean[cut_index:]
x_train_features = x_train_features[:cut_index]
y_train_valid = y_train_clean[:cut_index]

### normalize x
x_train_features_normalized = np.copy(x_train_features)
for i in range(len(x_train_features[0])):
    mean = np.nanmean(x_train_features[:, i])
    std = np.nanstd(x_train_features[:, i])
    x_train_features_normalized[:, i] = (x_train_features[:, i] - mean) / std

y_train_features_normalized = np.copy(y_train_valid)
for i in range(len(y_train_valid[0])):
    mean = np.nanmean(y_train_valid[:, i])
    std = np.nanstd(y_train_valid[:, i])
    y_train_features_normalized[:, i] = (y_train_valid[:, i] - mean) / std

x_validation_features_normalized = np.copy(x_validation_features)
for i in range(len(x_validation_features[0])):
    mean = np.nanmean(x_validation_features[:, i])
    std = np.nanstd(x_validation_features[:, i])
    x_validation_features_normalized[:, i] = (x_validation_features[:, i] - mean) / std

### regression
reg_lin = linear_model.LinearRegression().fit(x_train_features_normalized, y_train_valid)
y_validation_predicted_lin = reg_lin.predict(x_validation_features_normalized)
score_linear = r2_score(y_validation_predicted_lin, y_validation)
RMSE_linear = np.sqrt(np.mean((y_validation_predicted_lin - y_validation) ** 2))
RMSE_linear_rel = np.linalg.norm(y_validation_predicted_lin - y_validation, 'fro') / np.linalg.norm(y_validation, 'fro')

# reg_bay = linear_model.BayesianRidge().fit(x_train_features_normalized, y_train_valid[:, 0])
# y_validation_predicted_bay = reg_bay.predict(x_validation_features_normalized)
# score_bayes = r2_score(y_validation_predicted_bay, y_validation)
# RMSE_bayes = np.sqrt(np.mean((y_validation_predicted_bay - y_validation) ** 2))
# RMSE_bayes_rel = np.linalg.norm(y_validation_predicted_bay-y_validation, 'fro')/np.linalg.norm(y_validation, 'fro')

# reg_log = linear_model.LogisticRegression(max_iter=500).fit(x_train_features_normalized, y_train_valid[:, 0])
# y_validation_predicted_log = reg_log.predict(x_validation_features_normalized)
# score_log = r2_score(y_validation_predicted_log, y_validation)
# RMSE_log = np.sqrt(np.mean((y_validation_predicted_log - y_validation) ** 2))
# RMSE_log_rel = np.linalg.norm(y_validation_predicted_log-y_validation, 'fro')/np.linalg.norm(y_validation, 'fro')


# GPR
# kernel = RBF(length_scale=[1] * len(x_train_features_normalized.T),
#              length_scale_bounds=[[1, 100]] * len(x_train_features_normalized.T))+WhiteKernel(noise_level=0.373, noise_level_bounds=[0.1, 1])

kernel = ConstantKernel(0.1, (0.01, 10.0))*RBF(length_scale=[1] * len(x_train_features_normalized.T),
             length_scale_bounds=[[1, 100]] * len(x_train_features_normalized.T))+WhiteKernel(noise_level=0.373, noise_level_bounds=[0.3, 1])


gpr = GaussianProcessRegressor(kernel=kernel,
                               random_state=0, normalize_y=True).fit(x_train_features_normalized, y_train_valid)
y_validation_predicted_gpr = gpr.predict(x_validation_features_normalized)
y_train_predicted_gpr = gpr.predict(x_train_features_normalized)
score_gpr = r2_score(y_validation_predicted_gpr, y_validation)
RMSE_gpr = np.sqrt(np.mean((y_validation_predicted_gpr - y_validation) ** 2))
RMSE_gpr_rel = np.linalg.norm(y_validation_predicted_gpr - y_validation, 'fro') / np.linalg.norm(y_validation, 'fro')

# KL=gpr.kernel_.k1.length_scale
# mask=KL<20
# x_train_features_normalized_new=x_train_features_normalized[:, mask]
# x_validation_features_normalized_new=x_validation_features_normalized[:, mask]

# kernel_new = RBF(length_scale=[1] * len(x_train_features_normalized_new.T),
#              length_scale_bounds=[[10, 20]] * len(x_train_features_normalized_new.T))+WhiteKernel(noise_level=0.373, noise_level_bounds=[0.1, 1])
# gpr_new = GaussianProcessRegressor(kernel=kernel_new,
#                                random_state=0, normalize_y=True).fit(x_train_features_normalized_new, y_train_valid)
#
# y_validation_predicted_gpr_new = gpr_new.predict(x_validation_features_normalized_new)
# y_train_predicted_gpr_new = gpr_new.predict(x_train_features_normalized_new)
# score_gpr_new = r2_score(y_validation_predicted_gpr_new, y_validation)
# RMSE_gpr_new = np.sqrt(np.mean((y_validation_predicted_gpr_new - y_validation) ** 2))
# RMSE_gpr_rel_new = np.linalg.norm(y_validation_predicted_gpr_new - y_validation, 'fro') / np.linalg.norm(y_validation, 'fro')

print('LINEAR: score = ', score_linear, ' RMSE = ', RMSE_linear, ' rel RMSE = ', RMSE_linear_rel)
print('GPR: score = ', score_gpr, ' RMSE = ', RMSE_gpr, ' rel RMSE = ', RMSE_gpr_rel)
# print('NEW GPR: score = ', score_gpr_new, ' RMSE = ', RMSE_gpr_new, ' rel RMSE = ', RMSE_gpr_rel_new)

plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation[:, 0], c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation_predicted_gpr[:, 0], c='r',
         marker='o', markerfacecolor='None', linestyle='None')
plt.show()

plt.plot(np.linspace(0, y_train_valid.shape[0] - 1, y_train_valid.shape[0]), y_train_valid[:, 0], c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_train_valid.shape[0] - 1, y_train_valid.shape[0]), y_train_predicted_gpr[:, 0], c='g',
         marker='o', markerfacecolor='None', linestyle='None')
plt.show()

print('end')
#
# x_train_np = np.asarray(x_train_valid)
# x_train_np_norm = np.copy(x_train_np)
# for i in range(len(x_train_np[0])):
#     mean = np.nanmean(x_train_np[:, i])
#     std = np.nanstd(x_train_np[:, i])
#     x_train_np_norm[:, i] = (x_train_np[:, i] - mean) / std
#
# indecies = np.arange(len(x_train_np[0]))
# figure = plt.figure()
# for i in range(len(x_train_imputed) - 1200):
#     plt.plot(indecies, x_train_imputed[i, :])
# plt.show()
#
# indecies = np.arange(len(x_train_np_norm[0]))
# figure = plt.figure()
# for i in range(len(x_train_np_norm) - 1200):
#     plt.plot(indecies, x_train_np_norm[i, :])
# plt.show()
#
# print('end')
