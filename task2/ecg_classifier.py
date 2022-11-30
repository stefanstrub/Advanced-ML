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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel

from biosppy.signals import ecg
import time

file_path = 'task2/'
# file_path = ''
# x_test = pd.read_csv(file_path + 'X_test.csv')
# x_train = pd.read_csv(file_path + 'X_train.csv')
# y_train = pd.read_csv(file_path + 'y_train.csv')
# x_test.drop('id', inplace=True, axis=1)
# x_train.drop('id', inplace=True, axis=1)
# y_train.drop('id', inplace=True, axis=1)

# x_train_np = np.asarray(x_train)
# x_test_np = np.asarray(x_test)
# y_train_np = np.asarray(y_train)

# np.save(file_path + 'X_train', x_train_np)
# np.save(file_path + 'X_test', x_test_np)
# np.save(file_path + 'y_train', y_train_np)


x_train = np.load(file_path + 'X_train.npy')
x_test = np.load(file_path + 'X_test.npy')
y_train = np.load(file_path + 'y_train.npy')




x_train_shorted = []
for i in range(len(x_train)):
    x_train_shorted.append(x_train[i][~np.isnan(x_train[i])])
x_train_shorted = np.asarray(x_train_shorted)

### imputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# x_train_imputed = imputer.fit_transform(x_train)
# x_test_imputed = imputer.fit_transform(x_test)

start = time.time()
# mean_template = []
std_template = []
heart_rate = []
for i in range(len(x_train)):
    signal = np.asarray(x_train_shorted[i][100:])
    out = ecg.ecg(signal=signal, sampling_rate=300., show=False)
    # mean_template.append(np.mean(out['templates'], axis=0))
    std_template.append(np.std(out['templates'], axis=0))
    heart_rate.append(out['heart_rate'])
print('time',time.time() - start)

# np.save(file_path + 'mean_template', mean_template)
np.save(file_path + 'std_template', std_template)
np.save(file_path + 'heart_rate', heart_rate)

mean_templates = np.load(file_path + 'mean_template.npy')
std_templates = np.load(file_path + 'std_template.npy')
heart_rates = np.load(file_path + 'heart_rate.npy')

std_template[np.concatenate(y_train[:100]) == 3]
mean_template[np.concatenate(y_train[:100]) == 3]


i = 0
signal = np.asarray(x_train_shorted[i][100:])
out = ecg.ecg(signal=signal, sampling_rate=300., show=False)

signal_fd = []
for i in range(100):
    signal = mean_templates[i]
    signal_fd.append(np.fft.fft(signal))
signal_freq = np.fft.fftfreq(len(signal),1/300)

fig= plt.figure()
for i in range(100):
    plt.plot(signal_freq,signal_fd[i])
plt.show()

R_peaks = np.argmax(mean_templates[:,:90], axis=1)
mean_templates_reduced = mean_templates[R_peaks==60]
y_train_redued = y_train[R_peaks==60]

R_height = np.max(mean_templates_reduced[:,:90], axis=1)
T_height = np.max(mean_templates_reduced[:,90:], axis=1)
means = np.mean(mean_templates_reduced, axis=1)
S_height = np.min(mean_templates_reduced[:,60:], axis=1)
Q_height = np.min(mean_templates_reduced[:,:60], axis=1)
QRS_duration = np.argmin(mean_templates_reduced[:,60:], axis=1)+60 - np.argmin(mean_templates_reduced[:,:60], axis=1)
heart_rate_variation = []
heart_rate_minmax = []
for i in range(len(heart_rate)):
    heart_rate_variation.append(np.mean(np.abs(np.gradient(heart_rate[i]))))
    heart_rate_minmax.append(np.max(heart_rate[i]) - np.min(heart_rate[i]))

relative_minmaxdiff = (means - S_height) /(R_peaks - S_height)
mean_templates_class1_and_4 = mean_templates_reduced[relative_minmaxdiff < 0.2]
mean_templates_class2_and_3 = mean_templates_reduced[relative_minmaxdiff >= 0.2]
y_train_class1_and_4 = y_train_redued[relative_minmaxdiff < 0.2]

fig= plt.figure()
plt.plot(np.arange(len(x_train_shorted[R_peaks==25][0])),x_train_shorted[R_peaks==25][0])
plt.show()

fig = plt.figure()
for i in range(200):
    plt.plot(np.arange(len(mean_templates_reduced[i])), np.gradient(mean_templates_reduced[i]))
plt.show()

fig = plt.figure()
for i in range(200):
    plt.plot(np.arange(len(mean_templates_class1_and_4[i])), mean_templates_class1_and_4[i])
plt.show()
fig = plt.figure()
for i in range(200):
    plt.plot(np.arange(len(mean_templates_class1_and_4[i])), mean_templates_class2_and_3[i])
plt.show()

### outlier detection
pca = PCA(n_components=200)
x_reduced = pca.fit_transform(x_train_imputed)
GMM = GaussianMixture(n_components=1, random_state=0, reg_covar=1e-1).fit(x_reduced)
phi = -GMM.score_samples(x_reduced)

outlier_threshold=3440
outliers = x_train_imputed[phi > outlier_threshold, :]
x_train_clean = x_train_imputed[phi <= outlier_threshold, :]
y_train_clean = y_train_np[phi <= outlier_threshold, :]

print(outliers.shape[0]/x_train_imputed.shape[0])

idx=np.linspace(0,x_train.shape[0]-1,x_train.shape[0])
plt.plot(idx[phi>outlier_threshold], phi[phi>outlier_threshold], 'r*')
plt.plot(idx[phi<=outlier_threshold], phi[phi<=outlier_threshold], 'g*')
plt.show()

# pca_test = PCA(n_components=200)
x_reduced_test = pca.fit_transform(x_test_imputed)
phi_test = -GMM.score_samples(x_reduced_test)

x_test_clean = x_test_imputed[phi_test <= outlier_threshold, :]

idx=np.linspace(0,x_test.shape[0]-1,x_test.shape[0])
plt.plot(idx[phi_test>outlier_threshold], phi_test[phi_test>outlier_threshold], 'r*')
plt.plot(idx[phi_test<=outlier_threshold], phi_test[phi_test<=outlier_threshold], 'g*')
plt.show()

# outliers_fraction = 0.1
# clf = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf")
# clf.fit(x_train_imputed)
# outliers_prediction = clf.predict(x_train_imputed)
# outliers = x_train_imputed[outliers_prediction == -1]
# x_train_clean = x_train_imputed[outliers_prediction == 1]
# y_train_clean = y_train_np[outliers_prediction == 1]

#### feature detection
sel = VarianceThreshold(threshold=100)
sel.fit(x_train_clean)
x_train_features = sel.transform(x_train_clean)
x_test_features = sel.transform(x_test_clean)

selector = SelectKBest(f_regression, k=100)
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
x_test_features = x_test_features[:, selector.get_support()]

train_validation_ratio = 0.9999
# cut_index = int(len(x_train_features) * train_validation_ratio)
cut_index = int(len(x_train_features) - 10)
x_validation_features = x_train_features[cut_index:]
y_validation = y_train_clean[cut_index:]
x_train_features = x_train_features[:cut_index]
y_train_valid = y_train_clean[:cut_index]

### normalize x
x_train_features_normalized = np.copy(x_train_features)
x_validation_features_normalized = np.copy(x_validation_features)
x_test_features_normalized = np.copy(x_test_features)
for i in range(len(x_train_features[0])):
    mean = np.nanmean(x_train_features[:, i])
    std = np.nanstd(x_train_features[:, i])
    x_train_features_normalized[:, i] = (x_train_features[:, i] - mean) / std
    x_validation_features_normalized[:, i] = (x_validation_features[:, i] - mean) / std
    x_test_features_normalized[:, i] = (x_test_features[:, i] - mean) / std

### regression
reg_lin = linear_model.LinearRegression().fit(x_train_features_normalized, y_train_valid)
y_validation_predicted_lin = reg_lin.predict(x_validation_features_normalized)
score_linear = r2_score(y_validation_predicted_lin, y_validation)
RMSE_linear = np.sqrt(np.mean((y_validation_predicted_lin - y_validation) ** 2))
RMSE_linear_rel = np.linalg.norm(y_validation_predicted_lin - y_validation, 'fro') / np.linalg.norm(y_validation, 'fro')

reg_bay = linear_model.BayesianRidge().fit(x_train_features_normalized, y_train_valid[:, 0])
y_validation_predicted_bay = reg_bay.predict(x_validation_features_normalized)
y_train_predicted_bay = reg_bay.predict(x_train_features_normalized)
score_bayes = r2_score(y_validation_predicted_bay, y_validation)
RMSE_bayes = np.sqrt(np.mean((y_validation_predicted_bay - y_validation) ** 2))
RMSE_bayes_rel = np.linalg.norm(y_validation_predicted_bay - y_validation, 'fro') / np.linalg.norm(y_validation, 'fro')

# reg_log = linear_model.LogisticRegression(max_iter=500).fit(x_train_features_normalized, y_train_valid[:, 0])
# y_validation_predicted_log = reg_log.predict(x_validation_features_normalized)
# score_log = r2_score(y_validation_predicted_log, y_validation)
# RMSE_log = np.sqrt(np.mean((y_validation_predicted_log - y_validation) ** 2))
# RMSE_log_rel = np.linalg.norm(y_validation_predicted_log-y_validation, 'fro')/np.linalg.norm(y_validation, 'fro')


# GPR
kernel = RBF(length_scale=[10] * len(x_train_features_normalized.T),
             length_scale_bounds=[[1, 20]] * len(x_train_features_normalized.T)) + WhiteKernel(noise_level=0.2,
                                                                                                   noise_level_bounds=[
                                                                                                       0.1, 0.5])

gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True).fit(x_train_features_normalized,
                                                                                    y_train_valid)
#
y_validation_predicted_gpr = gpr.predict(x_validation_features_normalized)
y_train_predicted_gpr = gpr.predict(x_train_features_normalized)
score_gpr = r2_score(y_validation_predicted_gpr, y_validation)
RMSE_gpr = np.sqrt(np.mean((y_validation_predicted_gpr - y_validation) ** 2))
RMSE_gpr_rel = np.linalg.norm(y_validation_predicted_gpr - y_validation, 'fro') / np.linalg.norm(y_validation, 'fro')

y_test_predicted_gpr = gpr.predict(x_test_features_normalized)

print('LINEAR: score = ', score_linear, ' RMSE = ', RMSE_linear, ' rel RMSE = ', RMSE_linear_rel)
print('BAYESIAN: score = ', score_bayes, ' RMSE = ', RMSE_bayes, ' rel RMSE = ', RMSE_bayes_rel)
print('GPR: score = ', score_gpr, ' RMSE = ', RMSE_gpr, ' rel RMSE = ', RMSE_gpr_rel)

plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation[:, 0], c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation_predicted_gpr[:, 0], c='r',
         marker='o', markerfacecolor='None', linestyle='None')
plt.plot(np.linspace(0, y_validation.shape[0] - 1, y_validation.shape[0]), y_validation_predicted_bay, c='r',
         marker='o', markerfacecolor='None', linestyle='None')
plt.show()

plt.plot(np.linspace(0, y_train_valid.shape[0] - 1, y_train_valid.shape[0]), y_train_valid[:, 0], c='k', marker='*',
         linestyle='None')
plt.plot(np.linspace(0, y_train_valid.shape[0] - 1, y_train_valid.shape[0]), y_train_predicted_gpr, c='g', marker='o',
         markerfacecolor='None', linestyle='None')
plt.plot(np.linspace(0, y_train_valid.shape[0] - 1, y_train_valid.shape[0]), y_train_predicted_bay, c='g', marker='o',
         markerfacecolor='None', linestyle='None')
plt.show()

prediction = np.zeros((len(x_test),2))
prediction[:,0] = np.arange(len(x_test))
prediction[:,1] = y_test_predicted_gpr[:,0]
df = pd.DataFrame(prediction, columns=['id','y'])
df.to_csv('predictions.csv', index=False)

print('end')


