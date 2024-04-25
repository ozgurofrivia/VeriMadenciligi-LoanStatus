# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:59:13 2024

@author: ooozg
"""

# **Kütüphaneleri ekleme**
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# **Veri setini çekme, inceleme ve gereksiz sütunları çıkarma**
df = pd.read_csv('loan_data_1.csv')

print(df.head())

df.info()
print(df.isnull().sum())
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('Loan_ID', axis=1)

# **Cinsiyet sütunundaki boşlukları doldurmak için grafikleri karşılaştır**
#Cinsiyet
plt.figure(figsize=(9,5))
sns.countplot(x = df.Gender)
plt.show()
print(df.Gender.value_counts())

#Cinsiyet ile gelir arasında bir ilişki var mı?
plt.figure(figsize = (9,5))
sns.swarmplot(x = "Gender", y = "ApplicantIncome",data = df)
plt.legend(df.Gender.value_counts().index)
plt.title("Gender - Income")
plt.show()

#Kadınların gelir ortalaması
df_female = df[df.Gender == 'Female']
df_female_mean = df_female.ApplicantIncome.mean()

#Erkeklerin gelir ortalaması
df_male = df[df.Gender == 'Male']
df_male_mean = df_male.ApplicantIncome.mean()

print(f'Erkeklerin gelir ortalaması: {df_male_mean}\nKadınların gelir ortalaması: {df_female_mean}')

print('\n')
# #Cinsiyet ile kredi geçmişi arasındaki ilişki
df_Gender_CreditHistory = df.groupby(["Gender","Credit_History"]).size().reset_index(name = "Count")
df_Gender_CreditHistory['Credit_History'] = df_Gender_CreditHistory['Credit_History'].astype(str)
df_Gender_CreditHistory
plt.figure(figsize = (9,5))
sns.barplot(x = "Gender",y="Count", hue = "Credit_History",data = df_Gender_CreditHistory)
plt.title("Gender - Credit History")
plt.show()
#Kategorik veri olduğu için nan değerler mode kullanılarak doldurulacak
df.Gender = df.Gender.fillna('Male')
# **Ailedeki kişi sayısındaki boşlukları doldurmak için grafikleri karşılaştır**
#Ailedeki kişi sayısı
df.Dependents = df.Dependents.replace('3+', '3')

plt.figure(figsize=(9,5))
sns.countplot(x = df.Dependents )
plt.show()
print(df.Dependents .value_counts())

#Gelir ile ailedeki kişi sayısı arasındaki ilişki
plt.figure(figsize = (9,5))
sns.swarmplot(x = "Dependents", y = "ApplicantIncome",data = df)
plt.legend(df.Dependents.value_counts().index)
plt.title("Dependents - Income")
plt.show()
print('\n')

#Yaşanılan alan ile ailedeki kişi sayısı arasındaki ilişki
df_Dependents_PropertyArea = df.groupby(["Dependents","Property_Area"]).size().reset_index(name = "Count")
df_Dependents_PropertyArea['Dependents'] = df_Dependents_PropertyArea['Dependents'].astype(str)
df_Dependents_PropertyArea
plt.figure(figsize = (9,5))
sns.barplot(x = "Dependents",y="Count", hue = "Property_Area",data = df_Dependents_PropertyArea)
plt.title("Dependents - Property_Area")
plt.show()

#Grafiklerde büyük çoğunluk 0 olarak gözüküyor
df.Dependents = df.Dependents.fillna(0)

# **Eğitim sütunundaki boşlukları doldurmak için grafikleri karşılaştır**
#Eğitim
plt.figure(figsize=(9,5))
sns.countplot(x = df.Education)
plt.show()
print(df.Education.value_counts())

#Eğitim ile gelir arasındaki ilişki
plt.figure(figsize = (9,5))
sns.swarmplot(x = "Education", y = "ApplicantIncome",data = df)
plt.legend(df.Education.value_counts().index)
plt.title("Education - Income")
plt.show()

#Graduate gelir ortalaması
df_graduate = df[df.Education == 'Graduate']
df_graduate_mean = df_graduate.ApplicantIncome.mean()

#Not Graduate gelir ortalaması
df_notgraduate = df[df.Education == 'Not Graduate']
df_notgraduate_mean = df_notgraduate.ApplicantIncome.mean()

print(f'Graduate gelir ortalaması: {df_graduate_mean}\nNot graduate gelir ortalaması: {df_notgraduate_mean}')

print('\n')

# #Eğitim ile cinsiyet arasındaki ilişki
df_Education_Gender = df.groupby(["Education","Gender"]).size().reset_index(name = "Count")
df_Education_Gender
plt.figure(figsize = (9,5))
sns.barplot(x = "Education",y="Count", hue = "Gender",data = df_Education_Gender)
plt.title("Education - Gender")
plt.show()

#Kategorik veri olduğu için Education sütunundaki nan değerleri mode kullanarak doldur
df.Education = df.Education.fillna(df.Education.mode()[0])

# **Meslek sütunundaki boşlukları doldurmak için grafikleri karşılaştır**
#Meslek
plt.figure(figsize=(9,5))
sns.countplot(x = df.Self_Employed)
plt.show()
print(df.Self_Employed.value_counts())

#Meslek ile gelir arasındaki ilişki
plt.figure(figsize = (9,5))
sns.swarmplot(data=df, x="Self_Employed", y="ApplicantIncome")
plt.legend(df.Self_Employed.value_counts().index)
plt.title("Self Employed - Income")
plt.show

#Serbest çalışanların genel ortalaması
df_se_yes = df[df.Self_Employed == 'Yes']
df_se_yes_mean = df_se_yes.ApplicantIncome.mean()

#Serbest çalışmayanların gelir ortalaması
df_se_no = df[df.Self_Employed == 'No']
df_se_no_mean = df_se_no.ApplicantIncome.mean()

print(f'Serbest çalışmayanların gelir ortalaması:{df_se_no_mean}\nSerbest çalışanların gelir ortalaması:{df_se_yes_mean}')

#Gelirin serbest çalışıp çalışmama durumuna göre ortalamsına olan uzaklığını hesaplayıp hangisine daha yakınsa Yes veya No olarak doldurur
for index, row in df.iterrows():
    if pd.isnull(row['Self_Employed']):
        if abs(row['ApplicantIncome'] - df_se_yes_mean) <= abs(row['ApplicantIncome'] - df_se_no_mean):
            df.at[index, 'Self_Employed'] = 'Yes'
        else:
            df.at[index, 'Self_Employed'] = 'No'

# **Başvuran geliri sütunundaki boşlukları doldurmak için grafikleri karşılaştır**
#Gelir
plt.figure(figsize = (9,5))
sns.distplot(df.ApplicantIncome)
plt.show()
print(df.ApplicantIncome.describe())

#Genel gelir ortalaması
df_ApplicantIncome_mean = df.ApplicantIncome.mean()

print(f'Genel Gelir ortalaması: {df_ApplicantIncome_mean}\n')

#Serbest çalışıp çalışmadığına göre geliri ortalamayla doldurulacak.
df.loc[df['Self_Employed'] == 'Yes', 'ApplicantIncome'] = df[df['Self_Employed'] == 'Yes']['ApplicantIncome'].fillna(df_se_yes_mean)
df.loc[df['Self_Employed'] == 'No', 'ApplicantIncome'] = df[df['Self_Employed'] == 'No']['ApplicantIncome'].fillna(df_se_no_mean)
# **Kefil geliri sütunundaki boşlukları doldurmak için grafikleri karşılaştır**
#Kefil

plt.figure(figsize = (9,5))
sns.distplot(df.CoapplicantIncome)
plt.show()
print(df.CoapplicantIncome.describe())
print(df['CoapplicantIncome'].value_counts())

#Bu sütunu etkileyen bir değer olmadığı için genel ortalama ile doldurulacak
df.CoapplicantIncome = df.CoapplicantIncome.fillna(df.CoapplicantIncome.mean())
# **Kredi değeri sütunundaki boşlukları doldrurmak için grafikleri karşılaştır**
#Kredi Değeri
plt.figure(figsize = (9,5))
sns.distplot(df.LoanAmount)
plt.show()
print(df.LoanAmount.describe())

#Ortalama değerle doldurabiliriz
df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
# **Kredi süresi sütunundaki boşlukları doldrurmak için grafikleri karşılaştır**
#Kredi Süresi
plt.figure(figsize = (9,5))
sns.countplot(x = df.Loan_Amount_Term, )
plt.show()
print(df.Loan_Amount_Term.describe())
#Mode ile doldurulacak
df.Loan_Amount_Term = df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mode()[0])
# **Kredi geçmişi sütunundaki boşlukları doldrurmak için grafikleri karşılaştır**
#Kredi Geçmişi
plt.figure(figsize = (9,5))
sns.countplot(x = df.Credit_History)
plt.show()

# Cross-tabulation (çapraz tablo) oluşturma
cross_tab = pd.crosstab(df['Credit_History'], df['Loan_Status'], normalize='index')

#Kredi geçmişi ile kredi alma durumu arasındaki ilişki
plt.figure(figsize=(9, 5))
cross_tab.plot(kind='bar', stacked=True)
plt.xlabel("Credit_History")
plt.ylabel("Oran")
plt.title("Kredi Geçmişi ve Kredi Durumu İlişkisi (Oranlar)")
plt.show()

#Kredi almış olanların kredi geçmişi 1 almayanların 0 ile doldurulacak
df.loc[df['Loan_Status'] == 'Y', 'Credit_History'] = df[df['Loan_Status'] == 'Y']['Credit_History'].fillna(1)
df.loc[df['Loan_Status'] == 'N', 'Credit_History'] = df[df['Loan_Status'] == 'N']['Credit_History'].fillna(0)
#**Null değer kaldı mı?**
print(df.isnull().sum())
# **Kategorik verileri sayısal verilere dönüştürme**
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df.Married = le.fit_transform(df.Married)
df.Property_Area = le.fit_transform(df.Property_Area)
df.Loan_Status = le.fit_transform(df.Loan_Status)
df.Gender = le.fit_transform(df.Gender)
df.Self_Employed = le.fit_transform(df.Self_Employed)
df.Education = le.fit_transform(df.Education)
# **Kolerasyon tablosu ile sütunlar arasındaki ilişkiyi incele**
#Kolerasyon tablosu ile ilşki durumlarını incele
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='Blues', vmin=-1, vmax=1,)
plt.title('Korelasyon Matrisi')
plt.show()
# **Veriyi X ve y olarak ayır**


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# **Eğitim ve test seti olarak bölme**
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# **Normalizasyon yaparak büyük değerlerin baskınlığını azalt**
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# **Logistic Regression**

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred_lr = LR.predict(X_test)
print(np.concatenate((y_pred_lr.reshape(len(y_pred_lr), 1), y_test.reshape(len(y_test), 1)), 1))
# **Logistic Regression Confussion Matrix ve Accuracy Score**
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy_scores = {}
cm_lr = confusion_matrix(y_test, y_pred_lr)
ac_lr = accuracy_score(y_test, y_pred_lr, normalize = True)

accuracy_scores['Linear Regression'] = ac_lr
print(cm_lr)
print(ac_lr)
# **Logistic Regression için K-Fold Cross Validation**
from sklearn.model_selection import cross_val_score

accuracies_lr = cross_val_score(estimator = LR, X = X_train, y  = y_train, cv = 10)
kfold_accuracy_scores = {}
acm_lr = accuracies_lr.mean()
acstd_lr = accuracies_lr.std()
kfold_accuracy_scores['Logistic Regression'] = acm_lr
print(accuracies_lr.reshape(len(accuracies_lr),1),'\n',acm_lr,'\n',acstd_lr)
# **En iyi modeli ve en iyi parametreleri bulmak için Grid Search uygulanması**

from sklearn.model_selection import GridSearchCV
parameters = {
    'fit_intercept': [True, False],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(estimator = LR,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

LR2 = LogisticRegression(C = 0.1, fit_intercept  = True, penalty = 'l2')
LR2.fit(X_train, y_train)
y_pred_lr = LR2.predict(X_test)

h_accuracy_scores = {}
h_ac_lr = accuracy_score(y_test, y_pred_lr, normalize = True)
h_accuracy_scores['Linear Regression'] = h_ac_lr
print(h_ac_lr)
#**K-NN**
from sklearn.neighbors import KNeighborsClassifier

KNC = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
KNC.fit(X_train, y_train)
y_pred_knn = KNC.predict(X_test)
print(np.concatenate((y_pred_knn.reshape(len(y_pred_knn), 1), y_test.reshape(len(y_test), 1)), 1))
# **KNN Confussion Matrix ve Accuracy Score**
from sklearn.metrics import confusion_matrix, accuracy_score

cm_knc = confusion_matrix(y_test, y_pred_knn)
ac_knc = accuracy_score(y_test, y_pred_knn, normalize = True)
accuracy_scores['KNN'] = ac_knc
print(cm_knc)
print(ac_knc)
# **KNN için K-Fold Cross Validation**
from sklearn.model_selection import cross_val_score

accuracies_knc = cross_val_score(estimator = KNC, X = X_train, y  = y_train, cv = 10)
acm_knc = accuracies_knc.mean()
acstd_knc = accuracies_knc.std()
kfold_accuracy_scores['KNN'] = acm_knc
print(accuracies_knc.reshape(len(accuracies_knc),1),'\n',acm_knc,'\n',acstd_knc)
# **En iyi modeli ve en iyi parametreleri bulmak için Grid Search uygulanması**
from sklearn.model_selection import GridSearchCV
parameters = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}
grid_search = GridSearchCV(estimator = KNC,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

h_KNC = KNeighborsClassifier(metric = 'euclidean', n_neighbors = 13, weights = 'distance')
h_KNC.fit(X_train, y_train)
y_pred_knn = KNC.predict(X_test)

h_ac_knn = accuracy_score(y_test, y_pred_knn, normalize = True)
h_accuracy_scores['KNN'] = h_ac_knn
print(h_ac_knn)
# **SVC**
from sklearn.svm import SVC

svc = SVC(kernel = 'linear')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(np.concatenate((y_pred_svc.reshape(len(y_pred_svc), 1), y_test.reshape(len(y_test), 1)), 1))

# **SVC Confussion Matrix ve Accuracy Score**
from sklearn.metrics import confusion_matrix, accuracy_score
cm_svc = confusion_matrix(y_test, y_pred_svc)
ac_svc = accuracy_score(y_test, y_pred_svc, normalize = True)
accuracy_scores['SVC'] = ac_svc
print(cm_svc)
print(ac_svc)
# **SVC için K-Fold Cross Validation**
from sklearn.model_selection import cross_val_score

accuracies_svc = cross_val_score(estimator = svc, X = X_train, y  = y_train, cv = 10)
acm_svc = accuracies_svc.mean()
acstd_svc = accuracies_svc.std()
kfold_accuracy_scores['SVC'] = acm_svc
print(accuracies_svc.reshape(len(accuracies_svc),1),'\n',acm_svc,'\n',acstd_svc)
# **En iyi modeli ve en iyi parametreleri bulmak için Grid Search uygulanması**
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

h_svc = SVC(kernel = 'rbf', gamma = 0.1, C = 0.5, random_state = 0)
h_svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

h_ac_svc = accuracy_score(y_test, y_pred_svc, normalize = True)
h_accuracy_scores['SVC'] = h_ac_svc
print(h_ac_svc)
# **Decision Tree**
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier(criterion='entropy')
DTC.fit(X_train, y_train)
y_pred_dtc = DTC.predict(X_test)
print(np.concatenate((y_pred_dtc.reshape(len(y_pred_dtc), 1), y_test.reshape(len(y_test), 1)), 1))
# **Decision Tree Confussion Matrix ve Accuracy Score**
from sklearn.metrics import confusion_matrix, accuracy_score

cm_dtc = confusion_matrix(y_test, y_pred_dtc)
ac_dtc = accuracy_score(y_test, y_pred_dtc, normalize = True)
accuracy_scores['Decision Tree'] = ac_dtc
print(cm_dtc)
print(ac_dtc)

# **Decision Tree için K-Fold Cross Validation**
from sklearn.model_selection import cross_val_score

accuracies_dtc = cross_val_score(estimator = DTC, X = X_train, y  = y_train, cv = 10)
acm_dtc = accuracies_dtc.mean()
acstd_dtc = accuracies_dtc.std()
kfold_accuracy_scores['Decision Tree'] = acm_dtc
print(accuracies_dtc.reshape(len(accuracies_dtc),1),'\n',acm_dtc,'\n',acstd_dtc)
# **En iyi modeli ve en iyi parametreleri bulmak için Grid Search uygulanması**



from sklearn.model_selection import GridSearchCV

parameters = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_leaf': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator = DTC,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

h_DTC = DecisionTreeClassifier(criterion='entropy')
h_DTC.fit(X_train, y_train)
y_pred_dtc = h_DTC.predict(X_test)

h_ac_dtc = accuracy_score(y_test, y_pred_dtc, normalize = True)
h_accuracy_scores['Decision Tree'] = h_ac_dtc
print(h_ac_dtc)
# **Random Forest**
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RFC.fit(X_train, y_train)
y_pred_rfc = RFC.predict(X_test)
print(np.concatenate((y_pred_rfc.reshape(len(y_pred_rfc), 1), y_test.reshape(len(y_test), 1)), 1))
# **Random Forest Confussion Matrix ve Accuracy Score**
from sklearn.metrics import confusion_matrix, accuracy_score

cm_rfc = confusion_matrix(y_test, y_pred_rfc)
ac_rfc = accuracy_score(y_test, y_pred_rfc, normalize = True)
accuracy_scores['Random Forest'] = ac_rfc
print(cm_rfc)
print(ac_rfc)
# **Random Forest için K-Fold Cross Validation**
from sklearn.model_selection import cross_val_score

accuracies_rfc = cross_val_score(estimator = RFC, X = X_train, y  = y_train, cv = 10)
acm_rfc = accuracies_rfc.mean()
acstd_rfc = accuracies_rfc.std()
kfold_accuracy_scores['Random Forest'] = acm_rfc
print(accuracies_rfc.reshape(len(accuracies_rfc),1),'\n',acm_rfc,'\n',acstd_rfc)
# **En iyi modeli ve en iyi parametreleri bulmak için Grid Search uygulanması**
from sklearn.model_selection import GridSearchCV

parameters ={
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator = RFC,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

h_RFC = RandomForestClassifier(max_depth= 20, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 200)
h_RFC.fit(X_train, y_train)
y_pred_rfc = h_RFC.predict(X_test)

h_ac_rfc = accuracy_score(y_test, y_pred_rfc, normalize = True)
h_accuracy_scores['Random Forest'] = h_ac_rfc
print(h_ac_rfc)
# **Hyperparameter ayarından önceki doğruluk Skorları**
for algorithm, score in accuracy_scores.items():
    print(f'{algorithm:<25}{100*score:.2f}'+'%')
# **K-Fold Cross Validation doğruluk skorları**
for algorithm, score in kfold_accuracy_scores.items():
    print(f'{algorithm:<25}{100*score:.2f}%')
# **Hyperparameter ayarından sonraki doğruluk Skorları**
for algorithm, score in h_accuracy_scores.items():
    print(f'{algorithm:<25}{100*score:.2f}%')
