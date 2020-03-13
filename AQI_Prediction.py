# coding: utf-8

# In[9]:

# Imports several basic packages


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:

# ignore the warnings
import warnings

warnings.filterwarnings("ignore")

# import the package of heterogenous classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
# from hyperopt import tpe
# from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix

import xgboost as xgb
import lightgbm as lgb

# In[11]:

# input the data including train set, test set
train_set = pd.read_csv('/Users/zohar/Desktop/zzl/TrainDataSet.csv')
test_set = pd.read_csv('/Users/zohar/Desktop/zzl/TestDataSet.csv')

# In[4]:

print('There are', train_set.shape[0], "samples in the training set and", test_set.shape[0], "samples in the test set.")
print("In sum we have", train_set.shape[0] + test_set.shape[0], "samples.")

# preview the train set
print("Top View of train_set\n")
print(train_set.head())
print(train_set.info())
print('The shape of train_set:', train_set.shape)
print()

# preview the test set
print("Top View of test_set\n")
print(test_set.head())
print(test_set.info())
print('The shape of test_set:', test_set.shape)
print()

# In[5]:

# Several statistical information of train_set
train_set.describe()

# In[6]:

# Several statistical information of test_set
test_set.describe()

# In[7]:

# missing values in train_set
train_set.isnull().any()

# In[8]:

# missing values in test_set
test_set.isnull().any()

# In[11]:

# AQIRange visualization
plt.figure(figsize=(12, 8))
abalone_RingsRange = sns.countplot(x='AQI', data=train_set)
plt.title('Number of Air Station by Date with disparate AQIRange')
plt.xlabel('AQIRange')
plt.ylabel('Number of Air Station by Date')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/AQIRange visualization.png')
plt.show()

# show the exact number of each RingsRange in the histogram
# for k in abalone_RingsRange.patches:
#   abalone_RingsRange.annotate('{:.1f}'.format(int(k.get_height())), (k.get_x() + 0.1, k.get_height()))


# In[12]:

# AQIFCST visualization
# plot the distribution of AQIFCST
plt.figure(figsize=(12, 8))
sns.distplot(train_set['aqifcst'], color='steelblue')
plt.title('Distribution of Time Series Index')
plt.xlabel('Time Series Index(AQI FCST)')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Distribution of Time Series Index(AQI FCST).png')
plt.show()

# In[35]:

# poi_ratio visualization
# plot the distribution of poi_ratio
plt.figure(figsize=(12, 8))
sns.distplot(train_set['poi_ratio'], color='steelblue')
plt.title('Distribution of poi_ratio')
plt.xlabel('poi_ratio')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Distribution of poi_ratio.png')
plt.show()

# In[13]:

# Mean_Sea_Level_Pressure_daily_mean_MSL visualization
# plot the distribution of Mean_Sea_Level_Pressure_daily_mean_MSL
plt.figure(figsize=(12, 8))
sns.distplot(train_set['Mean_Sea_Level_Pressure_daily_mean_MSL'], color='steelblue')
plt.title('Distribution of Mean_Sea_Level_Pressure_daily_mean_MSL')
plt.xlabel('Mean_Sea_Level_Pressure_daily_mean_MSL')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Mean_Sea_Level_Pressure_daily_mean_MSL.png')
plt.show()

# In[14]:

# Temperature_daily_max_2m_above_gnd visualization
# plot the distribution of Temperature_daily_max_2m_above_gnd
plt.figure(figsize=(12, 8))
sns.distplot(train_set['Temperature_daily_max_2m_above_gnd'], color='steelblue')
plt.title('Distribution of Temperature_daily_max_2m_above_gnd')
plt.xlabel('Temperature_daily_max_2m_above_gnd')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Temperature_daily_max_2m_above_gnd.png')
plt.show()

# In[15]:

# Mean_Sea_Level_Pressure_daily_max_MSL visualization
# plot the distribution of TMean_Sea_Level_Pressure_daily_max_MSL
plt.figure(figsize=(12, 8))
sns.distplot(train_set['Mean_Sea_Level_Pressure_daily_max_MSL'], color='steelblue')
plt.title('Distribution of Mean_Sea_Level_Pressure_daily_max_MSL')
plt.xlabel('Mean_Sea_Level_Pressure_daily_max_MSL')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Mean_Sea_Level_Pressure_daily_max_MSL.png')
plt.show()

# In[16]:

# Mean_Sea_Level_Pressure_daily_min_MSL visualization
# plot the distribution of Mean_Sea_Level_Pressure_daily_min_MSL
plt.figure(figsize=(12, 8))
sns.distplot(train_set['Mean_Sea_Level_Pressure_daily_min_MSL'], color='steelblue')
plt.title('Distribution of Mean_Sea_Level_Pressure_daily_min_MSL')
plt.xlabel('Mean_Sea_Level_Pressure_daily_min_MSL')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Mean_Sea_Level_Pressure_daily_min_MSL.png')
plt.show()

# In[17]:

# Mean_Sea_Level_Pressure_daily_min_MSL visualization
# plot the distribution of Mean_Sea_Level_Pressure_daily_min_MSL
plt.figure(figsize=(12, 8))
sns.distplot(train_set['Mean_Sea_Level_Pressure_daily_min_MSL'], color='steelblue')
plt.title('Distribution of Mean_Sea_Level_Pressure_daily_min_MSL')
plt.xlabel('Mean_Sea_Level_Pressure_daily_min_MSL')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Mean_Sea_Level_Pressure_daily_min_MSL.png')
plt.show()

# In[18]:


# Wind_Speed_daily_min_10m_above_gnd visualization
# plot the distribution of Wind_Speed_daily_min_10m_above_gnd
plt.figure(figsize=(12, 8))
sns.distplot(train_set['Wind_Speed_daily_min_10m_above_gnd'], color='steelblue')
plt.title('Distribution of Wind_Speed_daily_min_10m_above_gnd')
plt.xlabel('Wind_Speed_daily_min_10m_above_gnd')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Wind_Speed_daily_min_10m_above_gnd.png')
plt.show()

# Total_Cloud_Cover_daily_min_sfc visualization
# plot the distribution of Total_Cloud_Cover_daily_min_sfc
plt.figure(figsize=(12, 8))
sns.distplot(train_set['Total_Cloud_Cover_daily_min_sfc'], color='steelblue')
plt.title('Distribution of Total_Cloud_Cover_daily_min_sfc')
plt.xlabel('Total_Cloud_Cover_daily_min_sfc')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/Total_Cloud_Cover_daily_min_sfc.png')
plt.show()

# tra_ratio visualization
# plot the distribution of tra_ratio
plt.figure(figsize=(12, 8))
sns.distplot(train_set['tra_ratio'], color='steelblue')
plt.title('Distribution of tra_ratio')
plt.xlabel('tra_ratio')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/tra_ratio.png')
plt.show()

# dbscan_ratio visualization
# plot the distribution of dbscan_ratio
plt.figure(figsize=(12, 8))
sns.distplot(train_set['dbscan_ratio'], color='steelblue')
plt.title('Distribution of dbscan_ratio')
plt.xlabel('dbscan_ratio')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/dbscan_ratio.png')
plt.show()

# diffbyratio visualization
# plot the distribution of diffbyratio
plt.figure(figsize=(12, 8))
sns.distplot(train_set['diffbyratio'], color='steelblue')
plt.title('Distribution of diffbyratio')
plt.xlabel('diffbyratio')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/diffbyratio.png')
plt.show()

# surr_effect visualization
# plot the distribution of surr_effect
plt.figure(figsize=(12, 8))
sns.distplot(train_set['surr_effect'], color='steelblue')
plt.title('Distribution of surr_effect')
plt.xlabel('surr_effect')
plt.ylabel('Nuclear density estimation')

# plt.savefig('/Users/zohar/Desktop/zzl/未命名文件夹/surr_effect.png')
plt.show()


# In[12]:

# Apply One-Hot-Encoding to categorical features
def One_hot_encoding(dataframe, features):
    feature_dummy = []
    for i in features:
        dataframe_dummy = pd.get_dummies(dataframe[i], prefix=i)
        feature_dummy.append(dataframe_dummy)
    dataframe_new = pd.concat(feature_dummy + [dataframe], axis=1)
    dataframe_new.drop(features, inplace=True, axis=1)
    return dataframe_new


# In[13]:

# input the data including train set, test set, countries set, age_gender set and the session set

# train_set = pd.read_csv('/Users/zhangzilin/Desktop/paper/DataSet/available/useful/TrainDataSetF.csv')
# test_set = pd.read_csv('/Users/zhangzilin/Desktop/paper/DataSet/available/useful/TestDataSetF.csv')


# In[14]:

# cut Shell_weight_range(g) into ranges 
# train_set['Shell_weight_range(g)'] = pd.cut(train_set['Shell weight(g)'], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
# test_set['Shell_weight_range(g)'] = pd.cut(test_set['Shell weight(g)'],  [0, 0.2, 0.4, 0.6, 0.8, 1.0])

# cut Shucked_weight_range(g) into ranges 
# train_set['Shucked_weight_range(g)'] = pd.cut(train_set['Shucked weight(g)'], [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75])
# test_set['Shucked_weight_range(g)'] = pd.cut(test_set['Shucked weight(g)'],  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75])

# cut Whole_weight_range(g) into ranges 
# train_set['Whole_weight_range(g)'] = pd.cut(train_set['Whole weight(g)'], [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# test_set['Whole_weight_range(g)'] = pd.cut(test_set['Whole weight(g)'], [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

# cut Diameter_range(mm) into ranges 
# train_set['Diameter_range(mm)'] = pd.cut(train_set['Diameter(mm)'], [0, 0.2, 0.4, 0.6, 0.8])
# test_set['Diameter_range(mm)'] = pd.cut(test_set['Diameter(mm)'], [0, 0.2, 0.4, 0.6, 0.8])


# In[15]:

# extract the label
label_cols = train_set['AQI']

# drop useless feature
drop_columns = ['ID']
train_set.drop(drop_columns, inplace = True, axis = 1)
test_set.drop(drop_columns, inplace = True, axis = 1)


# In[16]:

from sklearn.preprocessing import LabelEncoder

# apply one-hot encoding
feature_cols = []
# feature_cols = ['aqifcst','poi_ratio','Mean_Sea_Level_Pressure_daily_mean_MSL','Temperature_daily_max_2m_above_gnd','Mean_Sea_Level_Pressure_daily_max_MSL','Mean_Sea_Level_Pressure_daily_min_MSL','Total_Cloud_Cover_daily_min_sfc','Wind_Speed_daily_min_10m_above_gnd','tra_ratio','dbscan_ratio','diffbyratio','surr_effect']
train_set_oh = One_hot_encoding(train_set, feature_cols)
test_set_oh = One_hot_encoding(test_set, feature_cols)

# encode each country
y = label_cols
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = pd.DataFrame(data=encoder.transform(y), columns=['AQI'])

y_encoded_processed = []
for i in range(y_encoded.shape[0]):
    y_encoded_processed.append(y_encoded.values[i][0])

print(encoder.classes_)
print('The shape of train set is:\n', train_set_oh.shape)
print('The shape of test set is:\n', test_set_oh.shape)
print('The shape of label is:\n', y_encoded.shape)

# In[17]:

# drop label and save the processed train, test set
train_set_oh.drop('AQI', inplace=True, axis=1)
test_set_oh.drop('AQI', inplace=True, axis=1)

#train_set_oh.to_csv('/Users/zohar/Desktop/zzl/PROCESS/train_processed.csv', header=True, index=False)
#test_set_oh.to_csv('/Users/zohar/Desktop/zzl/PROCESS/test_processed.csv', header=True, index=False)

# In[18]:

# Feature importances using Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=300, random_state=0)
train_set_oh_feature = np.array(train_set_oh.columns)

forest.fit(train_set_oh, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print('Feature ranking:')

for i in range(train_set_oh.shape[1]):
    print(
        '%d. Feature %s (Relative Importance: %f)' % (i + 1, train_set_oh_feature[indices[i]], importances[indices[i]]))

# Plot the feature importances of the forest
plt.figure()
plt.title('Feature relative importances ranking')
plt.bar(range(train_set_oh.shape[1]), importances[indices], color='orange', align='center')
plt.xticks(range(train_set_oh.shape[1]), indices)
plt.xlim([-1, train_set_oh.shape[1]])

#plt.savefig('/Users/zohar/Desktop/zzl/PROCESS/Feature relative importances ranking.png')
plt.show()

# In[19]:

# Normalization processing to speed-up calculation
from sklearn import preprocessing

X_extracted = train_set_oh
numRows, numColumn = X_extracted.shape
X_scaled = preprocessing.scale(X_extracted)  # normalization processing to improve accuracy
class_names = ['A', 'B', 'C']

# In[20]:

# plot the confusion matrix with normalization or without normalization
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix after normalization')
    else:
        print('Confusion matrix without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Prediction')


# In[300]:
# # Logistic Regression
# logistic regression using 10-fold cross-validation
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial')
predict_total_log = cross_val_predict(classifier, X_scaled, y_encoded_processed, cv=10)

startTime_log = time.time()

Confusion_matrix_log = confusion_matrix(y_encoded, predict_total_log)
Confusion_matrix_log_dataframe = pd.DataFrame(data=Confusion_matrix_log,
                                              index=['A(true)', 'B(true)', 'C(true)'],
                                              columns=['A(prediction)', 'B(prediction)', 'C(prediction)', ])

print('Confusion matrix:')
print(Confusion_matrix_log_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_log, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of logistic regression')

#plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of logistic regression.png')
plt.show()

print('Accuracy score:')
accuracy_log = accuracy_score(y_encoded, predict_total_log)
print(accuracy_log)
print()

print('Precision score:')
precision_log = precision_score(y_encoded, predict_total_log, average='weighted')
print(precision_log)
print()

print('Recall score:')
recall_log = recall_score(y_encoded, predict_total_log, average='weighted')
print(recall_log)
print()

print('The report of classification using logistic regression: ')
print(classification_report(y_encoded, predict_total_log))

time_log = time.time() - startTime_log
print('The whole logistic regression takes %fs!' % time_log)

# In[301]:

# # Decision Tree
# decision tree using 10-fold cross-validation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# from hpsklearn import HyperoptEstimator, decision_tree

startTime_dt = time.time()

max_depth_dt = 10
min_samples_leaf_dt = 20
min_samples_split_dt = 6
classifier = DecisionTreeClassifier(criterion='gini', max_depth=max_depth_dt,
                                    min_samples_leaf=min_samples_leaf_dt,
                                    min_samples_split=min_samples_split_dt, random_state=0)
predict_total_dt = cross_val_predict(classifier, X_scaled, y_encoded_processed, cv=10)

Confusion_matrix_dt = confusion_matrix(y_encoded, predict_total_dt)
Confusion_matrix_dt_dataframe = pd.DataFrame(data=Confusion_matrix_dt,
                                             index=['A(true)', 'B(true)', 'C(true)'],
                                             columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_dt_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_dt, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of decision tree')
#plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of decision tree')
plt.show()

print('Accuracy score:')
accuracy_dt = accuracy_score(y_encoded, predict_total_dt)
print(accuracy_dt)
print()

print('Precision score:')
precision_dt = precision_score(y_encoded, predict_total_dt, average='weighted')
print(precision_dt)
print()

print('Recall score:')
recall_dt = recall_score(y_encoded, predict_total_dt, average='weighted')
print(recall_dt)
print()

print('The report of classification using decision tree: ')
print(classification_report(y_encoded, predict_total_dt))

time_dt = time.time() - startTime_dt
print('The whole decision tree takes %fs!' % time_dt)

# In[302]:
# # Naive Bayes
# Multinominal Naive Bayes
# using 10-fold cross-validation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

# from hpsklearn import HyperoptEstimator, multinomial_nb

startTime_nb = time.time()
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(train_set_oh)

# Multinomial Naive Bayes
classifier = MultinomialNB()
predict_total_nb = cross_val_predict(classifier, X_minmax, y_encoded_processed, cv=10)

Confusion_matrix_nb = confusion_matrix(y_encoded, predict_total_nb)
Confusion_matrix_nb_dataframe = pd.DataFrame(data=Confusion_matrix_nb,
                                             index=['A(true)', 'B(true)', 'C(true)'],
                                             columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_nb_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_nb, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of naive bayes')

plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of naive bayes.png')
plt.show()

print('Accuracy score:')
accuracy_nb = accuracy_score(y_encoded, predict_total_nb)
print(accuracy_nb)
print()

print('Precision score:')
precision_nb = precision_score(y_encoded, predict_total_nb, average='weighted')
print(precision_nb)
print()

print('Recall score:')
recall_nb = recall_score(y_encoded, predict_total_nb, average='weighted')
print(recall_nb)
print()

print('The report of classification using multinominal naive bayes: ')
print(classification_report(y_encoded, predict_total_nb))

time_nb = time.time() - startTime_nb
print('The whole naive bayes takes %fs!' % time_nb)

# In[ ]:

# Parameters tuning for random forest
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn import cross_validation, metrics

# determine the optimal parameters using 10-fold cross-validation
# n_estimators
# n_estimators_test = {'n_estimators': [100, 150, 200]}
# gsearch_n_estimators = GridSearchCV(estimator = RandomForestClassifier(), param_grid = n_estimators_test, cv = 10)
# gsearch_n_estimators.fit(X_scaled, y_encoded)
# n_estimators_optimal = gsearch_n_estimators.best_params_['n_estimators']
# print('The appropriate n_estimators is:', n_estimators_optimal)

# max_depth, min_samples_split, min_sample_leaf
# depth_split_leaf_test = {'max_depth': [20, 25, 30],
#                        'min_samples_split': [6, 8, 12],
#                        'min_samples_leaf':[20, 25, 30]}
# gsearch_depth_split_leaf = GridSearchCV(estimator = RandomForestClassifier(n_estimators = n_estimators_optimal),
#                                       param_grid = depth_split_leaf_test, scoring = 'recall', cv = 10)

# gsearch_depth_split_leaf.fit(X_scaled, y_encoded)
# max_depth_optimal = gsearch_depth_split_leaf.best_params_['max_depth']
# min_samples_split_optimal = gsearch_depth_split_leaf.best_params_['min_samples_split']
# min_samples_leaf_optimal = gsearch_depth_split_leaf.best_params_['min_samples_leaf']
# print('The appropriate max_depth is:', max_depth_optimal)
# print('The appropriate min_samples_split is:', min_samples_split_optimal)
# print('The appropriate min_samples_leaf is:', min_samples_leaf_optimal)

# n_jobs
# n_jobs_test = {'n_jobs': [20, 30]}
# gsearch_n_jobs =  GridSearchCV(estimator = RandomForestClassifier(n_estimators = n_estimators_optimal,
#                                                                 max_depth = max_depth_optimal,
#                                                                min_samples_split =  min_samples_split_optimal,
#                                                               min_samples_leaf = min_samples_leaf_optimal),
#                           param_grid = n_jobs_test, scoring = 'recall', cv = 10)
# gsearch_n_jobs.fit(X_scaled, y_encoded)
# n_jobs_optimal = gsearch_n_jobs.best_params_['n_jobs']
# print('The appropriate n_jobs is:', n_jobs_optimal)


# In[245]:

# Compute the optimal parameters for random forest
# from sklearn.model_selection import GridSearchCV
# tuned_params = [{'min_samples_split': [2, 3, 4],  'min_samples_leaf':[12, 14, 16], 'n_jobs': [10, 20, 30],
#                'n_estimators': [180, 200, 210], 'max_depth':[10, 12, 14]}]

# begin_t = time.time()
# model = RandomForestClassifier(random_state = 0)
# clf = GridSearchCV(estimator = model, param_grid = tuned_params, scoring = 'accuracy', cv = 10)

# y_encoded_processed = []
# for i in range(y_encoded.shape[0]):
#       y_encoded_processed.append(y_encoded.values[i][0])

# clf.fit(X_scaled, y_encoded_processed)
# end_t = time.time()

# print('Training time: ', round(end_t - begin_t, 3), 's')
# print('Current optimal parameters of random forest:', clf.best_params_)
# print(clf.best_estimator_)


# In[341]:


# Random Forest
# using 10-fold cross-validation
from sklearn.ensemble import RandomForestClassifier

# from hpsklearn import HyperoptEstimator, random_forest

startTime_rf = time.time()
n_estimators_rf = 30
n_jobs_rf = 12
max_depth_rf = 10
min_samples_leaf_rf = 4
min_samples_split_rf = 3
classifier = RandomForestClassifier(n_estimators=n_estimators_rf, n_jobs=n_jobs_rf,
                                    max_depth=max_depth_rf, min_samples_leaf=min_samples_leaf_rf,
                                    min_samples_split=min_samples_split_rf, random_state=0,
                                    bootstrap=True)
predict_total_rf = cross_val_predict(classifier, X_scaled, y_encoded_processed, cv=10)

Confusion_matrix_rf = confusion_matrix(y_encoded, predict_total_rf)
Confusion_matrix_rf_dataframe = pd.DataFrame(data=Confusion_matrix_rf,
                                             index=['A(true)', 'B(true)', 'C(true)'],
                                             columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_rf_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_rf, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of Random Forest')

plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of Random Forest.png')

plt.show()

print('Accuracy score:')
accuracy_rf = accuracy_score(y_encoded, predict_total_rf)
print(accuracy_rf)
print()

print('Precision score:')
precision_rf = precision_score(y_encoded, predict_total_rf, average='weighted')
print(precision_rf)
print()

print('Recall score:')
recall_rf = recall_score(y_encoded, predict_total_rf, average='weighted')
print(recall_rf)
print()

print('The report of classification using random forest: ')
print(classification_report(y_encoded, predict_total_rf))

time_rf = time.time() - startTime_rf
print('The whole random forest takes %fs!' % time_rf)

# # Support Vector Machine

# In[304]:

# Support Vector Machine
# using 10-fold cross-validation
from sklearn import svm

# from hpsklearn import HyperoptEstimator, svc


classifier = svm.SVC(kernel='linear', random_state=0)
predict_total_svm = cross_val_predict(classifier, X_scaled, y_encoded_processed, cv=10)

Confusion_matrix_svm = confusion_matrix(y_encoded, predict_total_svm)
Confusion_matrix_svm_dataframe = pd.DataFrame(data=Confusion_matrix_svm,
                                              index=['A(true)', 'B(true)', 'C(true)'],
                                              columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_svm_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_svm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of Support Vector Machine')
#plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of Support Vector Machine.png')
plt.show()

print('Accuracy score:')
accuracy_svm = accuracy_score(y_encoded, predict_total_svm)
print(accuracy_svm)
print()

print('Precision score:')
precision_svm = precision_score(y_encoded, predict_total_svm, average='weighted')
print(precision_svm)
print()

print('Recall score:')
recall_svm = recall_score(y_encoded, predict_total_svm, average='weighted')
print(recall_svm)
print()

print('The report of classification using support vector machine: ')
print(classification_report(y_encoded, predict_total_svm))

# In[246]:
# # XGBoost
# Compute the optimal parameters for xgboost
from xgboost.sklearn import XGBClassifier

tuned_params = [{'learning_rate': [0.05, 0.1, 0.15],
                 'n_estimators': [70, 80, 90], 'max_depth': [4, 6, 8],
                 'subsample': [0.6, 0.7, 0.8], 'colsample_bytree': [0.4, 0.5, 0.6]}]
begin_t = time.time()
clf = GridSearchCV(xgb.XGBClassifier(seed=7), tuned_params, scoring='accuracy', cv=10)

clf.fit(X_scaled, y_encoded_processed)
end_t = time.time()
print('Train time:  ', round(end_t - begin_t, 3), 's')
print('Current best parameters of xgboost: ', clf.best_params_)
print(clf.best_estimator_)

# In[331]:

# XGBoost
# using 10-fold cross-validation
from xgboost.sklearn import XGBClassifier
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

startTime_xgb = time.time()

classifier = XGBClassifier(objective='multi:softprob', learning_rate=0.1,
                           max_depth=3, n_estimators=60,
                           subsample=0.8, colsample_bytree=0.6, seed=0)

# Prediction using xgboost
predict_total_xgb = cross_val_predict(classifier, X_scaled, y_encoded_processed, cv=10)

Confusion_matrix_xgb = confusion_matrix(y_encoded, predict_total_xgb)
Confusion_matrix_xgb_dataframe = pd.DataFrame(data=Confusion_matrix_xgb,
                                              index=['A(true)', 'B(true)', 'C(true)'],
                                              columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_xgb_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_xgb, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of Xgboost')
plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of Xgboost.png')
plt.show()

print('Accuracy score:')
accuracy_xgb = accuracy_score(y_encoded, predict_total_xgb)
print(accuracy_xgb)
print()

print('Precision score:')
precision_xgb = precision_score(y_encoded, predict_total_xgb, average='weighted')
print(precision_xgb)
print()

print('Recall score:')
recall_xgb = recall_score(y_encoded, predict_total_xgb, average='weighted')
print(recall_xgb)
print()

print('The report of classification using xgboost: ')
print(classification_report(y_encoded, predict_total_xgb))

time_xgb = time.time() - startTime_xgb
print('The whole xgboost takes %fs!' % time_xgb)

# In[316]:
# # Light Gradient Boosting Machine
# Compute the optimal parameters for LightGBM
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

tuned_params = [{'learning_rate': [0.05, 0.1, 0.15],
                 'n_estimators': [80, 100, 120], 'max_depth': [3, 4, 5],
                 'min_child_samples': [30, 35, 40]}]

begin_t = time.time()
model = lgb.LGBMClassifier(objective='multiclass', seed=42)
clf = GridSearchCV(estimator=model, param_grid=tuned_params, scoring='accuracy', cv=10)

clf.fit(X_scaled, y_encoded_processed)
end_t = time.time()

print('Training time: ', round(end_t - begin_t, 3), 's')
print('Current optimal parameters of LightGBM: ', clf.best_params_)
print(clf.best_estimator_)

# In[317]:

# Light Gradient Boosting Machine
# using 10-fold cross-validation
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

startTime_lgb = time.time()

# LightGBM
# classifier = lgb.LGBMClassifier(boosting_type = 'gbdt', colsample_bytree = 1.0, learning_rate = 0.1,
#                               max_bin = 255, max_depth = 3, min_child_samples = 20, min_child_weight = 5,
#                              min_split_gain = 0.0, n_estimators = 80, objective = 'multiclass',
#                             random_state = 0, reg_alpha = 0.0, reg_lambda = 0.0, seed = 42,
#                            silent = True, subsample = 1.0,
#                           subsample_for_bin = 50000, subsample_freq = 1)

classifier = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                                importance_type='split', learning_rate=0.1, max_depth=3,
                                min_child_samples=40, min_child_weight=0.001, min_split_gain=0.0,
                                n_estimators=80, n_jobs=-1, num_leaves=31, objective='multiclass',
                                random_state=None, reg_alpha=0.0, reg_lambda=0.0, seed=42,
                                silent=True, subsample=1.0, subsample_for_bin=200000,
                                subsample_freq=0)

# cross_val_predict returns an array of the same size as `y` where each entry is a prediction obtained by cross validation:
predict_total_lgb = cross_val_predict(classifier, X_scaled, y_encoded_processed, cv=10)

Confusion_matrix_lgb = confusion_matrix(y_encoded, predict_total_lgb)
Confusion_matrix_lgb_dataframe = pd.DataFrame(data=Confusion_matrix_lgb,
                                              index=['A(true)', 'B(true)', 'C(true)'],
                                              columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_lgb_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_lgb, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of LightGBM')
# plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of LightGBM.png')

plt.show()

print('Accuracy score:')
accuracy_lgb = accuracy_score(y_encoded, predict_total_lgb)
print(accuracy_lgb)
print()

print('Precision score:')
precision_lgb = precision_score(y_encoded, predict_total_lgb, average='weighted')
print(precision_lgb)
print()

print('Recall score:')
recall_lgb = recall_score(y_encoded, predict_total_lgb, average='weighted')
print(recall_lgb)
print()

print('The report of classification using lightgbm: ')
print(classification_report(y_encoded, predict_total_lgb))

time_lgb = time.time() - startTime_lgb
print('The whole lightgbm takes %fs!' % time_lgb)

# In[6]:

# plot the ROC curve for different classifiers
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def ROC_curve_AUC(cf, X, y, fig_name):
    # Binarize the output
    y_bin = label_binarize(y, classes=[0, 1, 2])
    class_num = y_bin.shape[1]

    class_ringsrange = ['A', 'B', 'C']

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.1, random_state=0)

    if cf == 'Logistic Regression':
        classifier = OneVsRestClassifier(LogisticRegression(random_state=0))
        classifier.fit(X_train, y_train)
        proba = classifier.decision_function(X_test)
        y_bin_score = proba

    if cf == 'Random Forest':
        n_estimators_rf = 30
        n_jobs_rf = 12
        max_depth_rf = 10
        min_samples_leaf_rf = 4
        min_samples_split_rf = 3
        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators_rf, n_jobs=n_jobs_rf,
                                                                max_depth=max_depth_rf,
                                                                min_samples_leaf=min_samples_leaf_rf,
                                                                min_samples_split=min_samples_split_rf, random_state=0,
                                                                bootstrap=True))
        classifier.fit(X_train, y_train)
        y_bin_score = classifier.predict_proba(X_test)

    if cf == 'Support Vector Machine':
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=0))
        classifier.fit(X_train, y_train)
        y_bin_score = classifier.decision_function(X_test)

    if cf == 'Decision Tree':
        max_depth_dt = 10
        min_samples_leaf_dt = 20
        min_samples_split_dt = 6
        classifier = OneVsRestClassifier(DecisionTreeClassifier(criterion='gini', max_depth=max_depth_dt,
                                                                min_samples_leaf=min_samples_leaf_dt,
                                                                min_samples_split=min_samples_split_dt))
        classifier.fit(X_train, y_train)
        y_bin_score = classifier.predict_proba(X_test)

    if cf == 'Naive Bayes':
        classifier = OneVsRestClassifier(GaussianNB())
        classifier.fit(X_train, y_train)
        y_bin_score = classifier.predict_proba(X_test)

    if cf == 'Xgboost':
        classifier = OneVsRestClassifier(XGBClassifier(learning_rate=0.1,
                                                       max_depth=3, n_estimators=60,
                                                       subsample=0.8, colsample_bytree=0.6, seed=0))

        classifier.fit(X_train, y_train)
        y_bin_score = classifier.predict_proba(X_test)

    if cf == 'LightGBM':
        classifier = OneVsRestClassifier(
            lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                               importance_type='split', learning_rate=0.1, max_depth=3,
                               min_child_samples=40, min_child_weight=0.001, min_split_gain=0.0,
                               n_estimators=80, n_jobs=-1, num_leaves=31, objective='multiclass',
                               random_state=None, reg_alpha=0.0, reg_lambda=0.0, seed=42,
                               silent=True, subsample=1.0, subsample_for_bin=200000,
                               subsample_freq=0))
        classifier.fit(X_train, y_train)
        y_bin_score = classifier.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(class_num):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_bin_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= class_num

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr['macro'], tpr['macro'], label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc['macro']),
             color='navy', linestyle=':', linewidth=4)

    color_range = ['aqua', 'darkorange', 'cornflowerblue', 'tomato', 'violet', 'indigo',
                   'lime', 'orange', 'olive', 'saddlebrown', 'steelblue', 'pink']

    for i in range(class_num):
        plt.plot(fpr[i], tpr[i], lw=2, color=color_range[i],
                 label='ROC curve of class {0} (AUC = {1:0.2f})'''.format(class_ringsrange[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # plot the diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of multi-classes and macro-average ROC curve (classifier: %s)' % cf)
    plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.savefig(fig_name)
    plt.show()

    return fpr['macro'], tpr['macro'], roc_auc['macro']


# In[ ]:

# plot the roc curve of 12 classes with heterogenerous classifiers
''' 
    Classifiers are:
       1. 'Logistic Regression'
       2. 'Decision Tree'
       3. 'Naive Bayes'
       4. 'Random Forest'
       5. 'Support Vector Machine'
       6. 'XGBoost'
       7. 'LightGBM'
'''

# Logistic regression          
print('ROC curve and AUC using logistic regression:')
fpr_log, tpr_log, roc_auc_log = ROC_curve_AUC('Logistic Regression', X_scaled, y_encoded,
                                              '/Users/zohar/Desktop/zzl/ROC curve and AUC using logistic regression.png')

# Decision Tree
print('ROC curve and AUC using decision tree:')
fpr_dt, tpr_dt, roc_auc_dt = ROC_curve_AUC('Decision Tree', X_scaled, y_encoded,
                                           '/Users/zohar/Desktop/zzl/ROC curve and AUC using decision tree.png')

# Naive Bayes
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(train_set_oh)
print('ROC curve and AUC using naive bayes:')
fpr_nb, tpr_nb, roc_auc_nb = ROC_curve_AUC('Naive Bayes', X_minmax, y_encoded,
                                           '/Users/zohar/Desktop/zzl/ROC curve and AUC using naive bayes.png')

# Random Forest
print('ROC curve and AUC using random forest:')
fpr_rf, tpr_rf, roc_auc_rf = ROC_curve_AUC('Random Forest', X_scaled, y_encoded,
                                           '/Users/zohar/Desktop/zzl/ROC curve and AUC using random forest.png')

# Support Vector Machine
print('ROC curve and AUC using support vector machine:')
fpr_svm, tpr_svm, roc_auc_svm = ROC_curve_AUC('Support Vector Machine', X_scaled, y_encoded,
                                              '/Users/zohar/Desktop/zzl/ROC curve and AUC using support vector machine.png')

# Xgboost
print('ROC curve and AUC using xgboost:')
fpr_xgb, tpr_xgb, roc_auc_xgb = ROC_curve_AUC('Xgboost', X_scaled, y_encoded,
                                              '/Users/zohar/Desktop/zzl/ROC curve and AUC using xgboost.png')

# LightGBM      
print('ROC curve and AUC using light gradient boosting machine:')
fpr_lgb, tpr_lgb, roc_auc_lgb = ROC_curve_AUC('LightGBM', X_scaled, y_encoded,
                                              '/Users/zohar/Desktop/zzl/ROC curve and AUC using light gradient boosting machine.png')

color_range = ['aqua', 'darkorange', 'cornflowerblue', 'tomato', 'violet', 'indigo',
               'lime', 'orange', 'olive', 'saddlebrown', 'steelblue', 'pink']

# plot the macro-average ROC curve and AUC
plt.figure()
plt.plot(fpr_log, tpr_log, lw=2, color='lime', label='Logistic Regression (AUC = {0:0.2f})'''.format(roc_auc_log))
plt.plot(fpr_dt, tpr_dt, lw=2, color='orange', label='Decision Tree (AUC = {0:0.2f})'''.format(roc_auc_dt))
plt.plot(fpr_nb, tpr_nb, lw=2, color='aqua', label='Naive Bayes (AUC = {0:0.2f})'''.format(roc_auc_nb))
plt.plot(fpr_rf, tpr_rf, lw=2, color='cornflowerblue', label='Random Forest (AUC = {0:0.2f})'''.format(roc_auc_rf))
plt.plot(fpr_svm, tpr_svm, lw=2, color='tomato', label='Support Vector Machine (AUC = {0:0.2f})'''.format(roc_auc_svm))
plt.plot(fpr_xgb, tpr_xgb, lw=2, color='violet', label='XGBoost (AUC = {0:0.2f})'''.format(roc_auc_xgb))
# plt.plot(fpr_lgb, tpr_lgb, lw = 2, color = 'pink', label = 'LightGBM (AUC = {0:0.2f})'''.format(roc_auc_lgb))

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # plot the diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-average ROC curve of different classifiers')
plt.legend(loc="lower right")
plt.legend(bbox_to_anchor=(1.0, 1.0))
#plt.savefig('/Users/zohar/Desktop/zzl/Macro-average ROC curve of different classifiers.png')
plt.show()

# In[342]:
# # Stacking

from mlxtend.classifier import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# stacking strategy using xgboost, lighgbm, svm and random forest
# using 10-fold cross validation

# meta classifier is logistic regression
meta_clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=0)

# Xgboost
clf1 = XGBClassifier(objective='multi:softprob', learning_rate=0.1,
                           max_depth=3, n_estimators=60,
                           subsample=0.8, colsample_bytree=0.6, seed=0)

# LightGBM
clf2 = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                               importance_type='split', learning_rate=0.1, max_depth=3,
                               min_child_samples=40, min_child_weight=0.001, min_split_gain=0.0,
                               n_estimators=80, n_jobs=-1, num_leaves=31, objective='multiclass',
                               random_state=None, reg_alpha=0.0, reg_lambda=0.0, seed=42,
                               silent=True, subsample=1.0, subsample_for_bin=200000,
                               subsample_freq=0)

# Random Forest
n_estimators_rf = 30
n_jobs_rf = 12
max_depth_rf = 10
min_samples_leaf_rf = 4
min_samples_split_rf = 3
clf3 = RandomForestClassifier(n_estimators=n_estimators_rf, n_jobs=n_jobs_rf,
                              max_depth=max_depth_rf, min_samples_leaf=min_samples_leaf_rf,
                              min_samples_split=min_samples_split_rf, random_state=0,
                              bootstrap=True)

# SVM

clf4 = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=0))


startTime_stacking = time.time()

stacking_clf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4], meta_classifier=meta_clf)
predict_total_stacking = cross_val_predict(stacking_clf, X_scaled, y_encoded_processed, cv=10)

Confusion_matrix_stacking = confusion_matrix(y_encoded, predict_total_stacking)
Confusion_matrix_stacking_dataframe = pd.DataFrame(data=Confusion_matrix_stacking,
                                                   index=['A(true)', 'B(true)', 'C(true)'],
                                                   columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_stacking_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_stacking, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of stacking')
#plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of stacking.png')
plt.show()

print('Accuracy score:')
accuracy_stacking = accuracy_score(y_encoded, predict_total_stacking)
print(accuracy_stacking)
print()

print('Precision score:')
precision_stacking = precision_score(y_encoded, predict_total_stacking, average='weighted')
print(precision_stacking)
print()

print('Recall score:')
recall_stacking = recall_score(y_encoded, predict_total_stacking, average='weighted')
print(recall_stacking)
print()

print('The report of classification using stacking classifier: ')
print(classification_report(y_encoded, predict_total_stacking))

time_stacking = time.time() - startTime_stacking
print('The whole stacking classifier takes %fs!' % time_stacking)



# In[28]:
# # Voting
# Voting classifier including Xgboost, Random Forest, LightGBM, SVM
# weights are: Xgboost 6, Random Forest 3, LightGBM 2, SVM 1
# using 10-fold cross validation
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from mlxtend.classifier import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold

# Xgboost
clf1 = XGBClassifier(objective='multi:softprob', learning_rate=0.1,
                           max_depth=3, n_estimators=60,
                           subsample=0.8, colsample_bytree=0.6, seed=0)

# LightGBM
clf2 = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                               importance_type='split', learning_rate=0.1, max_depth=3,
                               min_child_samples=40, min_child_weight=0.001, min_split_gain=0.0,
                               n_estimators=80, n_jobs=-1, num_leaves=31, objective='multiclass',
                               random_state=None, reg_alpha=0.0, reg_lambda=0.0, seed=42,
                               silent=True, subsample=1.0, subsample_for_bin=200000,
                               subsample_freq=0)

# Random Forest
n_estimators_rf = 30
n_jobs_rf = 12
max_depth_rf = 10
min_samples_leaf_rf = 4
min_samples_split_rf = 3
clf3 = RandomForestClassifier(n_estimators=n_estimators_rf, n_jobs=n_jobs_rf,
                              max_depth=max_depth_rf, min_samples_leaf=min_samples_leaf_rf,
                              min_samples_split=min_samples_split_rf, random_state=0,
                              bootstrap=True)

# SVM

clf4 = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=0,probability=True))


# Voting classifier
voting_clf = VotingClassifier(estimators=[('xgb', clf1), ('lgb', clf2), ('rf', clf3), ('svm', clf4)],
                              voting='soft', weights=[6, 2, 3, 1])

startTime_vote = time.time()

predict_total_vote = cross_val_predict(voting_clf, X_scaled, y_encoded_processed, cv=10)

Confusion_matrix_vote = confusion_matrix(y_encoded, predict_total_vote)
Confusion_matrix_vote_dataframe = pd.DataFrame(data=Confusion_matrix_vote,
                                               index=['A(true)', 'B(true)', 'C(true)'],
                                               columns=['A(prediction)', 'B(prediction)', 'C(prediction)'])
print('Confusion matrix:')
print(Confusion_matrix_vote_dataframe)
print()

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(Confusion_matrix_vote, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of voting')
plt.savefig('/Users/zohar/Desktop/zzl/Normalized confusion matrix of voting.png')

plt.show()

print('Accuracy score:')
accuracy_vote = accuracy_score(y_encoded, predict_total_vote)
print(accuracy_vote)
print()

print('Precision score:')
precision_vote = precision_score(y_encoded, predict_total_vote, average='weighted')
print(precision_vote)
print()

print('Recall score:')
recall_vote = recall_score(y_encoded, predict_total_vote, average='weighted')
print(recall_vote)
print()

print('The report of classification using voting classifier: ')
print(classification_report(y_encoded, predict_total_vote))

time_vote = time.time() - startTime_vote
print('The whole voting classifier takes %fs!' % time_vote)

# # Final Training and Prediction

# In[21]:

# input the data including train set, test set, countries set, age_gender set and the session set
train_set = pd.read_csv('//Users/zohar/Desktop/zzl/TrainDataSet.csv')
test_set = pd.read_csv('//Users/zohar/Desktop/zzl/TestDataSet.csv')

train_set_processed = pd.read_csv('/Users/zohar/Desktop/zzl/PROCESS/train_processed.csv')
test_set_processed = pd.read_csv('/Users/zohar/Desktop/zzl/PROCESS/test_processed.csv')

train_set_processed.index = train_set['ID']
test_set_processed.index = test_set['ID']

# normalization the train and test set together
train_test_processed = pd.concat((train_set_processed, test_set_processed), axis=0)
train_test_scaled = preprocessing.scale(train_test_processed)  # normalization processing to improve accuracy
train_test_processed_new = pd.DataFrame(train_test_scaled, index=train_test_processed.index)

train_set_processed_scaled = train_test_processed_new.ix[train_set_processed.index]
test_set_processed_scaled = train_test_processed_new.ix[test_set_processed.index]

# In[247]:

# Compute the optimal parameters of random forest using 10-fold cross validation
from sklearn.model_selection import GridSearchCV

tuned_params = [{'min_samples_split': [2, 3, 4], 'min_samples_leaf': [12, 15, 18], 'n_jobs': [15, 20, 25],
                 'n_estimators': [25, 30, 40], 'max_depth': [3, 4, 5]}]

begin_t = time.time()
model = RandomForestClassifier(random_state=0)
clf = GridSearchCV(estimator=model, param_grid=tuned_params, scoring='accuracy', cv=10)

clf.fit(train_set_processed_scaled, y_encoded_processed)
end_t = time.time()

print('Training time: ', round(end_t - begin_t, 3), 's')
print('Current optimal parameters of random forest:', clf.best_params_)
print(clf.best_estimator_)

# In[248]:

# Compute the optimal parameters of xgboost using 10-fold cross validation
from xgboost.sklearn import XGBClassifier

tuned_params = [{'learning_rate': [0.05, 0.1, 0.15],
                 'n_estimators': [70, 80, 60], 'max_depth': [3, 4, 5],
                 'subsample': [0.4, 0.8, 0.6], 'colsample_bytree': [0.5, 0.6, 0.7]}]
begin_t = time.time()
clf = GridSearchCV(xgb.XGBClassifier(seed=7), tuned_params, scoring='accuracy', cv=10)

clf.fit(train_set_processed_scaled, y_encoded_processed)
end_t = time.time()
print('Train time: ', round(end_t - begin_t, 3), 's')
print('Current best parameters of xgboost:', clf.best_params_)
print(clf.best_estimator_)

# In[347]:

# Compute the optimal parameters of lightgbm using 10-fold cross validation
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

tuned_params = [{'learning_rate': [0.05, 0.1, 0.15],
                 'n_estimators': [90, 100, 110], 'max_depth': [3, 4, 5],
                 'min_child_samples': [25, 30, 35]}]

begin_t = time.time()
model = lgb.LGBMClassifier(objective='multiclass', seed=42)
clf = GridSearchCV(estimator=model, param_grid=tuned_params, scoring='accuracy', cv=10)

clf.fit(train_set_processed_scaled, y_encoded_processed)
end_t = time.time()

print('Training time: ', round(end_t - begin_t, 3), 's')
print('Current optimal parameters of lightGBM:', clf.best_params_)
print(clf.best_estimator_)

# In[39]:

# Voting classifier including Xgboost, Random Forest, LightGBM,SVM
# weights are: Xgboost 6, Random Forest 3, LightGBM 2, SVM 1
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# Xgboost
clf1 = XGBClassifier(objective='multi:softprob', learning_rate=0.1,
                     max_depth=3, n_estimators=60,
                     subsample=0.8, colsample_bytree=0.6, seed=0)

# LightGBM
clf3 = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                          importance_type='split', learning_rate=0.05, max_depth=3,
                          min_child_samples=35, min_child_weight=0.001, min_split_gain=0.0,
                          n_estimators=90, n_jobs=-1, num_leaves=31, objective='multiclass',
                          random_state=None, reg_alpha=0.0, reg_lambda=0.0, seed=42,
                          silent=True, subsample=1.0, subsample_for_bin=200000,
                          subsample_freq=0)

# Logistic Regression
clf2 = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=0,probability=True))

# Random Forest
n_estimators_rf = 80
n_jobs_rf = 15
max_depth_rf = 7
min_samples_leaf_rf = 15
min_samples_split_rf = 2
clf4 = RandomForestClassifier(n_estimators=n_estimators_rf, n_jobs=n_jobs_rf,
                              max_depth=max_depth_rf, min_samples_leaf=min_samples_leaf_rf,
                              min_samples_split=min_samples_split_rf, random_state=0,
                              bootstrap=True)

# Random Forest
n_estimators_rf = 80
n_jobs_rf = 20
max_depth_rf = 9
min_samples_leaf_rf = 12
min_samples_split_rf = 2
clf3 = RandomForestClassifier(n_estimators=n_estimators_rf, n_jobs=n_jobs_rf,
                              max_depth=max_depth_rf, min_samples_leaf=min_samples_leaf_rf,
                              min_samples_split=min_samples_split_rf, random_state=0,
                              bootstrap=True)

# Voting classifier
voting_clf = VotingClassifier(estimators=[('xgb', clf1), ('SVM', clf2), ('lgb', clf3), ('rf', clf4)],
                              voting='soft', weights=[6, 1, 2, 3])

startTime_vote = time.time()
voting_clf.fit(train_set_processed_scaled, y_encoded)
predict_total_vote = pd.DataFrame(voting_clf.predict(test_set_processed_scaled))

# save the prediction results
AQI = encoder.inverse_transform(predict_total_vote)
final_prediction = pd.DataFrame(data=AQI, index=test_set['ID'], columns=['AQI'])
final_prediction.to_csv('/Users/zohar/Desktop/zzl/final_prediction.csv')
