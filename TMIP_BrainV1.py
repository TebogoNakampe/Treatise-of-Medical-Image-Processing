
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sqlalchemy import create_engine
get_ipython().magic('matplotlib inline')
pd.set_option("display.max_columns",101)

tumour_features = pd.read_excel('VASARI_MRI_features (gmdi - wiki).xls')
tumour_features.head()

# Rename columns with meaningful names and drop ID and comments columns

cols = ['Sample','Radiologist','Tumor_Location','Side_of_Tumor_Epicenter','Eloquent_Brain','Enhancement_Quality',
        'Proportion_Enhancing','Proportion_nCET', 'Proportion_Necrosis','Cyst','Multifocal_or_Multicentric',
        'T1_FLAIR_Ratio','Thickness_Enhancing_Margin','Definition_Enhancing_Margin', 'Definition_Non_Enhancing_Margin',
        'Proportion_of_Edema','Edema_Crosses_Midline','Hemorrhage','Diffusion','Pial_Invasion','Ependymal_Invasion',
        'Cortical_Involvement','Deep_WM_Invasion','nCET_Tumor_Crosses_Midline','Enhancing_Tumor_Crosses_Midline',
        'Satellites','Calvarial_Remodeling','Extent_Resection_Enhancing_Tumor','Extent_Resection_nCET',
        'Extent_Resection_Vasogenic_Edema','Lesion_Size_x','Lesion_Size_y']

tumour_features = tumour_features.drop(['ID', 'comments'], axis=1)
tumour_features.columns = cols
tumour_features.head()

# Read in CSV and format dataframe

patients_info = pd.read_excel('clinical_2014-01-16.xlsx')
patients_info.drop(patients_info.columns[6:], axis=1, inplace=True)
patients_info.columns = ['Sample','Age','Gender','Survival_months','Disease','Grade']
patients_info.head()

patients_info['Disease'].value_counts()

# Calculate baseline accuracy ofthe whole dataset (for astrocytoma/gbm/oligodendroglioma diagnosis only)
baseline_whole = float(patients_info['Disease'].value_counts()[0])/float(np.sum([patients_info['Disease'].value_counts()[i] for i in range(3)]))
print('Baseline accuracy of the whole dataset:', np.round(baseline_whole*100, 2), '%')

# Merge the two dataframes
tumours = pd.merge(patients_info[['Sample','Disease','Survival_months']], tumour_features, on='Sample', how='inner')

# Assign survival column type to int
tumours['Survival_months'] = tumours['Survival_months'].astype('int')

# Display first index of each disease

print('Astocytoma first index:', tumours.loc[tumours['Disease']==' ASTROCYTOMA', 'Disease'].index[0])
print('GBM first index:', tumours.loc[tumours['Disease']==' GBM', 'Disease'].index[0])
print('Oligodendroglioma first index:', tumours.loc[tumours['Disease']==' OLIGODENDROGLIOMA', 'Disease'].index[0])

# Encode Disease column
from sklearn.preprocessing import LabelEncoder

tumours['Disease'] = LabelEncoder().fit_transform(tumours['Disease'])

tumours.head()

# Display encoded numbers associated with each disease

print('Astocytoma:', tumours.iloc[0, 1])
print('GBM:', tumours.iloc[3, 1])
print('Oligodendroglioma:', tumours.iloc[21, 1])
# Create a new dataframe with only one line per patient: 
# value is assigned to the most frequent score between the three radiologists 
# or if there are three different values to score from radiologist #2

dicty = {}
for col in tumours.columns[4:]:
    dicty[col]=[]
    for sample in tumours['Sample'].unique():
        count = tumours.loc[tumours['Sample']==sample, col].value_counts().sort_values(ascending=False)
        if len(count) == 2:
            dicty[col].append(count.index[0])
        else:
            dicty[col].append(tumours.loc[(tumours['Sample']==sample) & (tumours['Radiologist']=='Radiologist #2'), col].values[0])

target = tumours.iloc[range(0,96,3), 0:3].reset_index()            
tumours_clean = target.join(pd.DataFrame(dicty))
tumours_clean.drop(['index'], axis=1, inplace=True)
tumours_clean.to_csv('tumours_target_features.csv')#ABint
tumours_clean.head()

# Read clean csv file back in
tumours = pd.read_csv('tumours_target_features.csv', encoding='UTF8',index_col=0)

# Plot histogramme of tumour type in dataset (32 patients)

plt.hist(tumours['Disease'])
plt.xticks([0,1,2],['Astrocytoma', 'GBM', 'Oligodendroglioma'])
plt.xlabel('Tumour type')
plt.ylabel('Count')
plt.title('Tumour types occurence in dataset of 32 patients')
plt.show()

# Plot relations between variables and display tumour types

sns.pairplot(tumours, hue='Disease', hue_order=[0,1,2], vars=['Survival_months',
                                            'Eloquent_Brain','Proportion_Enhancing','Proportion_nCET',
                                            'Proportion_Necrosis','Multifocal_or_Multicentric','T1_FLAIR_Ratio',
                                            'Proportion_of_Edema','Extent_Resection_Enhancing_Tumor',
                                            'Extent_Resection_nCET','Extent_Resection_Vasogenic_Edema',
                                            'Lesion_Size_x', 'Lesion_Size_y'])
plt.legend()
plt.show()

#Plot correlation heatmap

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(tumours.corr())

from sklearn.preprocessing import MinMaxScaler

X = tumours.drop(['Sample', 'Disease', 'Survival_months', 'Tumor_Location'], axis=1)
X_cols = X.columns
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X, columns=X_cols)
X.head()
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)

pd.DataFrame(np.cumsum([round(i*100, 2) for i in pca.explained_variance_ratio_]), index=range(1,30), columns=['Cumulative % of Variance Explained']).head(10)
X_pca = PCA(n_components=2).fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=tumours['Disease'], cmap='rainbow')
plt.xlabel('pca 1')
plt.ylabel('pca 2')
plt.title('Tumour types scatter against pca components')
plt.show()
y = tumours['Disease']
print('Baseline accuracy:', y.value_counts()[1]/float(len(y)))
# Store all model scores in a list of tuples to compare them
model_scores = []
gs_logreg = GridSearchCV(logreg, {'C': [10e-5, 10e-4, 10e-3, 10e-2, 0.1, 1, 10, 100, 10e3], 'penalty':['l1','l2']})
gs_logreg.fit(X,y)

print('Regression parameters:', gs_logreg.best_params_, '\n')

print('Cross validated accuracy logistic regression C=10, Ridge penalty (optimised):', gs_logreg.best_score_)

model_scores.append(('gs_logreg', gs_logreg.best_score_))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y, cross_val_predict(gs_logreg.best_estimator_, X, y, cv=3)))
print(classification_report(y, cross_val_predict(gs_logreg.best_estimator_, X, y, cv=3)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

print('Cross validated accuracy scores KNN 5 neighbours:', cross_val_score(knn, X, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(knn, X, y, cv=3).mean())
gs_knn = GridSearchCV(knn, {'n_neighbors': range(1, 11)})
gs_knn.fit(X,y)

print('KNN parameters:', gs_knn.best_params_, '\n')

print('Cross validated accuracy KNN 6 neighbours:', gs_knn.best_score_)

model_scores.append(('gs_knn', gs_knn.best_score_))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
rfc = RandomForestClassifier()

print('Cross validated accuracy scores random forest:', cross_val_score(rfc, X, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(rfc, X, y, cv=3).mean())
gs_rfc = GridSearchCV(rfc, {'n_estimators': range(5, 85, 5)})
gs_rfc.fit(X,y)

print('Random forest parameters:', gs_rfc.best_params_, '\n')

print('Cross validated accuracy Random Forest 15 estimators:', gs_rfc.best_score_)

model_scores.append(('gs_rfc', gs_rfc.best_score_))
print(confusion_matrix(y, cross_val_predict(gs_rfc.best_estimator_, X, y, cv=3)))
print(classification_report(y, cross_val_predict(gs_rfc.best_estimator_, X, y, cv=3)))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(X, y)
lda.explained_variance_ratio_
plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='rainbow')
plt.xlabel('lda1')
plt.ylabel('lda2')
plt.show()
print('Cross validated accuracy scores LDA:', cross_val_score(lda, X, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(lda, X, y, cv=3).mean())

model_scores.append(('LDA',cross_val_score(lda, X, y, cv=3).mean()))
print(confusion_matrix(y, cross_val_predict(lda, X, y, cv=3)))
print(classification_report(y, cross_val_predict(lda, X, y, cv=3)))
from sklearn.naive_bayes import MultinomialNB

nbm = MultinomialNB()

print('Cross validated accuracy scores Naive Bayes:', cross_val_score(nbm, X, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(nbm, X, y, cv=3).mean())

model_scores.append(('NBm', cross_val_score(nbm, X, y, cv=3).mean()))
print(confusion_matrix(y, cross_val_predict(nbm, X, y, cv=3)))
print(classification_report(y, cross_val_predict(nbm, X, y, cv=3)))
print('Model scores', model_scores)
rfc_opt = gs_rfc.best_estimator_
# Feature importance in Random Forest model
rfc_top10 = pd.DataFrame({'columns':X.columns, 
                      'feature_importances_rfc':rfc_opt.feature_importances_}).sort_values('feature_importances_rfc', 
                                                                                            ascending=False).head(10)
rfc_top10
X_3cols = X[['Proportion_nCET', 'Thickness_Enhancing_Margin', 'Enhancement_Quality']]
print('Cross validated accuracy scores logistic regression on RF 3 columns:', cross_val_score(gs_logreg.best_estimator_, X_3cols, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(gs_logreg.best_estimator_, X_3cols, y, cv=3).mean())

model_scores.append(('Logreg_rf3', cross_val_score(gs_logreg.best_estimator_, X_3cols, y, cv=3).mean()))
print('Cross validated accuracy scores knn on RF 3 columns:', cross_val_score(gs_knn.best_estimator_, X_3cols, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(gs_knn.best_estimator_, X_3cols, y, cv=3).mean())

model_scores.append(('KNN_rf3', cross_val_score(gs_knn.best_estimator_, X_3cols, y, cv=3).mean()))
print('Cross validated accuracy scores random forest on 3 most important features:', cross_val_score(rfc_opt, X_3cols, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(rfc_opt, X_3cols, y, cv=3).mean())

model_scores.append(('rfc_rf3', cross_val_score(rfc_opt, X_3cols, y, cv=3).mean()))
model_scores.sort(key=lambda tup: tup[1])

plt.bar(range(8), [tup[1] for tup in model_scores], align='center')
plt.xticks(range(8), [tup[0] for tup in model_scores])
plt.show()

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=5)
selected_data = selector.fit_transform(X, y)

kbest_columns = X.columns[selector.get_support()]
Xtbest = pd.DataFrame(selected_data, columns=kbest_columns)

# p-values of each feature in SelectKBest
SelectKBest_top10 = pd.DataFrame({'columns':X.columns, 'p_values':selector.pvalues_}).sort_values('p_values').head(10)
SelectKBest_top10

# Plot correlations between selected 5 features
sns.heatmap(tumours[kbest_columns].corr())
print('Cross validated accuracy scores logistic regression on KBest columns:', cross_val_score(gs_logreg.best_estimator_, Xtbest, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(gs_logreg.best_estimator_, Xtbest, y, cv=3).mean())

model_scores.append(('Logreg_kbest', cross_val_score(gs_logreg.best_estimator_, Xtbest, y, cv=3).mean()))
print('Cross validated accuracy scores knn on KBest columns:', cross_val_score(gs_knn.best_estimator_, Xtbest, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(gs_knn.best_estimator_, Xtbest, y, cv=3).mean())

model_scores.append(('KNN_kbest', cross_val_score(gs_knn.best_estimator_, Xtbest, y, cv=3).mean()))
print('Cross validated accuracy scores random forest on KBest columns:', cross_val_score(gs_rfc.best_estimator_, Xtbest, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(gs_rfc.best_estimator_, Xtbest, y, cv=3).mean())

model_scores.append(('rfc_kbest', cross_val_score(gs_rfc.best_estimator_, Xtbest, y, cv=3).mean()))
model_scores.sort(key=lambda tup: tup[1])

plt.bar(range(11), [tup[1] for tup in model_scores],color="green")
plt.xticks(range(11), [tup[0] for tup in model_scores], rotation=45)
plt.show()
from sklearn.feature_selection import RFE

# Features rank in logistic regression RFE
rfe_logreg = RFE(gs_logreg.best_estimator_, n_features_to_select=1)
rfe_logreg.fit(X, y)

RFE_logreg_top10 = pd.DataFrame({'columns':X.columns, 
                              'ranking_RFE_logreg':rfe_logreg.ranking_}).sort_values('ranking_RFE_logreg').head(10)
RFE_logreg_top10
# Define X with only 5 bets features
rfe_logreg = RFE(gs_logreg.best_estimator_, n_features_to_select=5)
X_rfe_logreg = rfe_logreg.fit_transform(X, y)
rfe_logreg_columns = X.columns[rfe_logreg.get_support()]
X_rfe_logreg = pd.DataFrame(X_rfe_logreg, columns=rfe_logreg_columns)
print('Cross validated accuracy scores RFE logistic regression:', cross_val_score(rfe_logreg, X, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(rfe_logreg, X, y, cv=3).mean())

model_scores.append(('logreg_rfe', cross_val_score(rfe_logreg, X, y, cv=3).mean()))
# Features rank in random forest RFE
rfe_rf = RFE(rfc_opt, n_features_to_select=1)
rfe_rf.fit(X, y)

RFE_rfc_top10 = pd.DataFrame({'columns':X.columns, 
                              'ranking_RFE_rfc':rfe_rf.ranking_}).sort_values('ranking_RFE_rfc').head(10)
RFE_rfc_top10
# Define X with only 5 best features
rfe_rf = RFE(rfc_opt, n_features_to_select=5)
X_rfe_rf = rfe_rf.fit_transform(X, y)

rfe_rf_columns = X.columns[rfe_rf.get_support()]
X_rfe_rf = pd.DataFrame(X_rfe_rf, columns=rfe_rf_columns)
print('Cross validated accuracy scores RFE random forest:', cross_val_score(rfe_rf, X, y, cv=3))
print('Mean cross validated accuracy:', cross_val_score(rfe_rf, X, y, cv=3).mean())

model_scores.append(('rfc_rfe', cross_val_score(rfe_rf, X, y, cv=3).mean()))
# Feature importances according to the three techniques: Random Forest, SelectKBest and RFE on Random Forest
pd.DataFrame({'rfc_top10':rfc_top10['columns'].values, 
              'SelectKBest_top10':SelectKBest_top10['columns'].values,
              'RFE_logreg_top10':RFE_logreg_top10['columns'].values,
              'RFE_rfc_top10':RFE_rfc_top10['columns'].values})
model_scores.sort(key=lambda tup: tup[1])

plt.bar(range(13), [tup[1] for tup in model_scores], color="green")
plt.xticks(range(13), [tup[0] for tup in model_scores], rotation=45)
plt.show()
acc_cv_mean = cross_val_score(rfc_opt, Xtbest, y, cv=3).mean()
acc_cv_std = cross_val_score(rfc_opt, Xtbest, y, cv=3).std()

print('Accuracy:', round(acc_cv_mean,4), '+/-', round(acc_cv_std,4))
print('Predicted probabilities of tumour types:')
pd.DataFrame(cross_val_predict(gs_rfc.best_estimator_, Xtbest, y, cv=3, method='predict_proba'), 
             columns=['0 - Astrocytoma', '1 - GBM', '2 - Oligodendroglioma']).join(tumours['Disease'])
print('Confusion matrix')
pd.DataFrame(confusion_matrix(y, cross_val_predict(gs_rfc.best_estimator_, Xtbest, y, cv=3)), 
             columns = ['predicted_Astrocytoma', 'predicted_GBM', 'predicted_Oligodendroglioma'],
             index = ['actual_Astrocytoma', 'actual_GBM', 'actual_Oligodendroglioma'])
print('Classification report')
print(classification_report(y, cross_val_predict(gs_rfc.best_estimator_, Xtbest, y, cv=3)))
# Plot ROC curve of GBM vs non-GBM (two classes output instead of three)

y_bin = [1 if i==1 else 0 for i in y]
y_bin_score = cross_val_predict(gs_rfc.best_estimator_, Xtbest, y_bin, cv=3, method='predict_proba')[:,1]

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_bin, y_bin_score)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='g',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='b', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Random Forest (n=25) on 5 best features')
plt.legend(loc="lower right")
plt.show()
# Plot precision-recall curve of GBM vs non-GBM

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_bin, y_bin_score)
average_precision = average_precision_score(y_bin, y_bin_score)

plt.plot(recall, precision, color='navy', 
         label='Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall  - Random Forest (n=25) on 5 best features')
plt.legend(loc="lower left")
plt.show()


