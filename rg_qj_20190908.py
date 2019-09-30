#!/usr/local/bin python3.6

# -*- coding: utf-8 -*-

"""

@time: 2019-09-08 12:47

@Author: Qianjia Jiang

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from datetime import datetime
from math import sqrt, ceil

import warnings

from scipy import stats
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=Warning)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut

# import models
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor

STARTIME = datetime.now()

# load data

path = 'Y:\\protein_structure\\_QSAR_file\\HP_column\\saltc03_QSAR\KA\\6_logo\\'
df_data = pd.read_excel('Y:\protein_structure\_QSAR_file\HP_column\descriptor_amber14eht.xlsx',
                        sheet_name='qsar_data_salt03',
                        index_col=0)
print(df_data)

np.random.seed(0)

X = df_data.values[:, :-2]
y = df_data.values[:, -2]


##calculate pearson correlation, sort values and plot
fig1 = plt.subplots()
df_corr = df_data.corrwith(df_data['SMA_KA_c1'])
# df_corr = df_data.corrwith(df_data['SMA_NU_c1'])
df_corr_sort = df_corr.sort_values(ascending=False)
df_corr_sort.drop(index=['SMA_KA_c1'], inplace=True)
df_corr_sort.drop(index=['SMA_NU_c1'], inplace=True)
df_corr_sort.to_excel(path+'correlation_analysis.xlsx')
df_corr_sort.plot.bar()
plt.xticks(fontsize=7)
plt.title('Correlation analysis of protein properties and Ka', pad=8, fontweight='bold')
plt.tight_layout()
plt.savefig(path+'correlation analysis.png', dpi=400)

fig2 = plt.subplots(figsize=(8, 8))
df_corr2 = df_data.iloc[:, :-2].corr()
sns.heatmap(df_corr2, vmin=-1, vmax=1, cmap='bwr', xticklabels=1, yticklabels=1)
plt.title('Correlation between different descriptors', pad=8, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(path+'descriptors_correlation analysis.png', dpi=400)


class ModelClass:
    def __init__(self, model_name, estimator, scoring='r2'):
        self.model_name = model_name
        self.estimator = estimator
        self.scoring = scoring


def print_time(simstarttime):
    """
    Helper function to print more legible time output
    :param simstarttime:
    :return:
    """

    def trim_time(time):
        if time.split(":")[0] != "0":
            time = time.split(".")[0] + " (h:mm:ss)"
        elif int(time.split(":")[1]) > 9:
            time = time[2:-5] + " (mm:ss.u)"
        elif int(time.split(":")[1]) > 0:
            time = time[3:-5] + " (m:ss.u)"
        elif int(time[5:-7]) > 0:
            time = time[5:-5] + " (ss.u)"
        elif int(time[5:-7]) == 0:
            time = time[-6:] + " (us)"
        return time

    sim_time = str(datetime.now() - simstarttime)
    sim_time = trim_time(sim_time)
    print(f"This took {sim_time}")


def compare_fitted_params(estim, model_class):
    """
    Prints a message detailing the fitted best_params_ hyperparameters
    for all estimators in the estimators collection

    :param estimators:
    :param model_class:
    """
    message = f"{model_class.model_name}: \n"
    msg_part = estim.named_steps['gridsearchcv'].best_params_
    message += f"{msg_part} \n"
    return message


def compare_elasticnetcv_params(estim, model_class):
    message = f"{model_class.model_name}: \n"
    alpha = estim.named_steps["elasticnetcv"].alpha_
    l1_ratio = estim.named_steps["elasticnetcv"].l1_ratio_
    message += f"'log(alpha)': {round(np.log10(alpha), 1)},  'l1_ratio': {l1_ratio} \n"
    return message


def plot_arrangement_from_n(n):
    """ Helper function to find the best arrangement for n subplots """

    def smalles_perimiter_rectangle(number):
        a = factors(number)[-1]
        return a, int(number / a)

    def factors(number):
        return [factor for factor in range(1, int(number ** 0.5) + 1) if number % factor == 0]

    def next_square_number(x):
        return np.square(ceil(sqrt(x)))

    rectangles = [smalles_perimiter_rectangle(m) for m in range(n, next_square_number(n) + 1)]
    smallest = min(rectangles, key=lambda x: sum(x))
    return smallest


groups = df_data.groupby(['patch_hyd']).grouper.group_info[0]


def train_test(model, X, y, groups):
    """
    do cross validation and get the cv_results
    :param model: model after hyperparameter tuning
    :param X: array like,training x
    :param y: array like, training y
    :param score: estimated score, could be "r2",'neg_mean_absolute_error','neg_mean_squared_error'
    :return:
    """
    ytest = np.array([])
    ypred = np.array([])
    kfold = LeaveOneGroupOut()
    for idx, (train_index, test_index) in enumerate(kfold.split(X, y, groups=groups)):
        x_train = X[train_index]
        y_train = y[train_index]
        reg = model.estimator.fit(x_train, y_train)
        y_test = y[test_index]
        x_test = X[test_index]
        y_pred = reg.predict(x_test)
        ytest = np.append(ytest, y_test)
        ypred = np.append(ypred, y_pred)
    estimator_name = model.model_name
    test_score = r2_score(ytest, ypred)
    r = stats.pearsonr(ytest, ypred)
    fitted_estimator = model.estimator.fit(X, y)

    return estimator_name, test_score, r, ytest, ypred, fitted_estimator


models = list()
cv_inner = 6
models.append(ModelClass('PLSRegressor', make_pipeline(StandardScaler(), PLSRegression())))

models.append(ModelClass('PLSRegression 2D',
                         make_pipeline(StandardScaler(), PCA(n_components=0.95),
                                       PolynomialFeatures(2, interaction_only=True, include_bias=True),
                                       PLSRegression())))

models.append(ModelClass('LinearRegressor', make_pipeline(StandardScaler(), LinearRegression())))

models.append(ModelClass('HuberRegressor', make_pipeline(StandardScaler(), HuberRegressor())))

models.append(ModelClass('Lars', make_pipeline(LarsCV(cv=cv_inner, normalize=True))))

models.append(ModelClass('LassoLarsCV', LassoLarsCV(cv=cv_inner, normalize=True)))

models.append(ModelClass('LassoLarsIC', make_pipeline(LassoLarsIC())))

models.append(ModelClass('BayesianRidge', make_pipeline(StandardScaler(), BayesianRidge())))

models.append(
    ModelClass('ElasticNet kBest std', make_pipeline(StandardScaler(), SelectKBest(mutual_info_regression, k=6),
                                                     ElasticNetCV(l1_ratio=[.05, .1, .2, .5, .7, .9, .95, .99, 1],
                                                                  cv=cv_inner,
                                                                  normalize=False, max_iter=90000))))

models.append(ModelClass('ElasticNet kBest norm', make_pipeline(SelectKBest(mutual_info_regression, k=6),
                                                                ElasticNetCV(
                                                                    l1_ratio=[.05, .1, .2, .5, .7, .9, .95, .99, 1],
                                                                    cv=cv_inner,
                                                                    normalize=True, max_iter=90000))))

models.append(ModelClass('ElasticNet 1D norm',
                         make_pipeline(
                             ElasticNetCV(l1_ratio=[.05, .1, .2, .5, .7, .9, .95, .99, 1], cv=cv_inner,
                                          normalize=True, max_iter=90000))))

models.append(ModelClass('PCA ElasticNet 1D norm',
                         make_pipeline(PCA(n_components=0.95),
                                       PolynomialFeatures(1, interaction_only=False, include_bias=True),
                                       ElasticNetCV(l1_ratio=[.05, .1, .2, .5, .7, .9, .95, .99, 1],
                                                    cv=cv_inner,
                                                    normalize=True, max_iter=90000))))

models.append(ModelClass('PCA ElasticNet 2D norm',
                         make_pipeline(StandardScaler(), PCA(n_components=0.95),
                                       PolynomialFeatures(2, interaction_only=True, include_bias=True),
                                       ElasticNetCV(l1_ratio=[.05, .1, .2, .5, .7, .9, .95, .99, 1],
                                                    cv=cv_inner,
                                                    normalize=True, max_iter=9000))))

models.append(ModelClass("SGDRegressor", make_pipeline(StandardScaler(),
                                                       GridSearchCV(SGDRegressor(),
                                                                    param_grid={'alpha': np.logspace(-7, -1, num=4),
                                                                                'loss': ['squared_loss', 'huber'],
                                                                                'epsilon': [0.2, 0.5, 0.9]},
                                                                    cv=cv_inner))))

models.append(ModelClass('KNeighbors', make_pipeline(StandardScaler(),
                                                     GridSearchCV(KNeighborsRegressor(),
                                                                  param_grid={'weights': ['uniform', 'distance']},
                                                                  cv=cv_inner))))

gp_kernel_1 = ExpSineSquared(length_scale=0.001, periodicity=145, length_scale_bounds=(1e-5, 1),
                             periodicity_bounds=(1e-2, 1e5)) + WhiteKernel(1e0)
gp_kernel_2 = ExpSineSquared(length_scale=0.005, periodicity=50.0,
                             periodicity_bounds=(1e0, 1e5)) + WhiteKernel(1e-3) + RBF()
gp_kernel_3 = RBF() + WhiteKernel(1e-3)
models.append(ModelClass('GaussianProcess_multi_kernel', make_pipeline(StandardScaler(),
                                                                       GridSearchCV(GaussianProcessRegressor(),
                                                                                    param_grid={'kernel': [gp_kernel_1,
                                                                                                           gp_kernel_2,
                                                                                                           gp_kernel_3]},
                                                                                    cv=cv_inner))))

models.append(ModelClass('GaussianProcess_multi_kernel_PCA',
                         make_pipeline(StandardScaler(), PCA(n_components=0.95),
                                       GaussianProcessRegressor(kernel=gp_kernel_3))))

models.append(ModelClass('SVR', make_pipeline(StandardScaler(),
                                              GridSearchCV(SVR(),
                                                           param_grid={'kernel': ['rbf', 'linear', 'sigmoid'],
                                                                       'C': np.logspace(-5, 10, num=16, base=2),
                                                                       'gamma': np.logspace(-2, 2, 5),
                                                                       'epsilon': [0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0]},
                                                           cv=cv_inner))))

models.append(ModelClass('DecisionTree', make_pipeline(StandardScaler(),
                                                       GridSearchCV(DecisionTreeRegressor(),
                                                                    param_grid={
                                                                        'min_samples_leaf': np.linspace(0.01, 0.5, 10)},
                                                                    cv=cv_inner))))

models.append(ModelClass('tree_AdaboostRegressor', make_pipeline(StandardScaler(),
                                                                 GridSearchCV(AdaBoostRegressor(),
                                                                              param_grid={
                                                                                  'n_estimators': [100, 300, 500],
                                                                                  'learning_rate': [0.01, 0.1, 1],
                                                                                  'loss': ['linear', 'square',
                                                                                           'exponential']},
                                                                              cv=cv_inner))))

models.append(ModelClass('tree_BaggingRegressor', make_pipeline(StandardScaler(),
                                                                GridSearchCV(BaggingRegressor(),
                                                                             param_grid={
                                                                                 'n_estimators': [100, 300, 500],
                                                                                 'max_features': np.arange(0.1, 1.1,
                                                                                                           10)},
                                                                             cv=cv_inner))))

models.append(ModelClass('RandomForest', make_pipeline(StandardScaler(),
                                                       GridSearchCV(RandomForestRegressor(),
                                                                    param_grid={'max_features': np.arange(0.1, 1.1, 10),
                                                                                'n_estimators': [100, 300, 500, 1000]},
                                                                    cv=cv_inner))))

models.append(ModelClass('GradientBoostingRegressor', make_pipeline(StandardScaler(),
                                                                    GridSearchCV(GradientBoostingRegressor(),
                                                                                 param_grid={'loss': ['ls', 'quantile'],
                                                                                             'n_estimators': [50, 100,
                                                                                                              300]},
                                                                                 cv=cv_inner))))

models.append(ModelClass("ANN_large",
                         make_pipeline(StandardScaler(),
                                       MLPRegressor(hidden_layer_sizes=(50, 20, 20, 10),
                                                    activation="tanh",
                                                    solver="lbfgs", max_iter=5000))))

models.append(ModelClass("ANN_small",
                         make_pipeline(StandardScaler(),
                                       GridSearchCV(MLPRegressor(hidden_layer_sizes=(50), max_iter=9000),
                                                    param_grid={'activation': ['tanh', 'relu'],
                                                                'solver': ['lbfgs', 'adam']},
                                                    cv=cv_inner))))

models.append(ModelClass("ANN_linear",
                         make_pipeline(StandardScaler(),
                                       MLPRegressor(hidden_layer_sizes=(30, 20, 10),
                                                    solver="lbfgs", max_iter=90000))))

##save all cross validation results in a list
##save best parameters in a file
##sort cv results by mean test score
model_train_times = []
results = list()

with open(path+'paramter_compararion.txt', 'w') as file:
    for model in models:
        train_starttime = datetime.now()
        cv_results = train_test(model, X, y, groups)
        train_duration = datetime.now() - train_starttime
        print(f"completed model {model.model_name} after {train_duration}")
        model_train_times.append(train_duration)

        results.append(cv_results)
        if "GridSearchCV" in str(model.estimator):
            file.write(compare_fitted_params(cv_results[-1], model))
        if "ElasticNetCV" in str(model.estimator):
            file.write(compare_elasticnetcv_params(cv_results[-1], model))
    file.close()
resultslist = list(results)
sort_result = sorted(resultslist, key=itemgetter(1), reverse=True)
names, test_score, r, ytest, ypred, fitted_estimator = zip(*sort_result)

# save the train and test score for checking
df = pd.DataFrame(sort_result)
df = df.iloc[:, :-1]
df_new = df.rename(
    columns={0: 'Regressor', 1: 'test_score', 2: 'pcc', 3: 'ytest', 4: 'ypred'})
df_new.to_excel(path+'cv_results.xlsx', index=False)

labels = names
x_pos = np.arange(len(labels))

## plot test score
fig = plt.figure(figsize=(8, 8))
plt.bar(x_pos, test_score, align='center', alpha=0.5, capsize=4)
plt.xticks(ticks=x_pos, labels=labels, rotation=90, fontsize=7)
plt.ylabel(r'Test Score ($R^2$)')
plt.title('Comparison of regressors', fontweight='bold')
plt.tight_layout()
plt.savefig(path+'comparison of regressors (test).png', dpi=400)

i = 0
fig, ax = plt.subplots(*plot_arrangement_from_n(len(models)), figsize=(20, 12))
ax = ax.flatten()
for names, test_score, r, ytest, ypred, fitted_estimator in sort_result:
    ax[i].scatter(ytest, ypred, s=10)
    ax[i].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax[i].set_xlabel('Measured',fontsize=7)
    ax[i].set_ylabel('Predicted',fontsize=7)
    ax[i].set_title(names+"\n"+f'($R^2$='+str(test_score)+')',fontsize=9)
    i = i + 1
plt.tight_layout()
plt.savefig(path+'Cross validate plot.png', dpi=400)



fig = plt.figure(figsize=(8,8))
df_plot = df_new[df_new['test_score'] >= 0]
df_plot = df_plot.iloc[:,:2]
df_plot.plot.bar(x='Regressor',y='test_score',legend=False)
plt.xticks(rotation=90, fontsize=7)
plt.ylabel(r'Test Score ($R^2$)')
plt.title('Comparison of regressors', fontweight='bold')
plt.tight_layout()
plt.savefig(path+'comparison of positive regressors.png', dpi=400)


print(print_time(STARTIME))


