
"""
Yapacagımız adımlar

1-Gerekli kütüphanelerin indirilemsi
2-Bizden istenilen veriyi yüzeysel inceleme (data import)
3-Kayıp verileri düzeltcez
4-Verimizi detaylıca inceleme
5-Aykırı verilerin çıkaraılması
6-Feature Enginering yapacagız
7-Verimizi ayırcaz test -train
8-Verimizi standartlaşma yapacığız
9-Verimizi ML algoritmlarıyla eğitecez
10- En iyi paremetreleri bulup vsonuçları inceliyeceğiz

"""




import pandas as pd
import numpy as np
import sns
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection, metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor,LocalOutlierFactor,NeighborhoodComponentsAnalysis,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import  neighbors
from sklearn.svm import SVR
from warnings import filterwarnings
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import norm, skew, describe
import xgboost as xgb

filterwarnings("ignore")



testnamecolumns =["Pclass","Sex","Age"]
trainnamecolumns =["Survived","Pclass","Sex","Age"]

silinecekler =['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

dataTrain =pd.read_csv("train.csv")
dataTest =pd.read_csv("test.csv")
dataTest =dataTest.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
dataTrain =dataTrain.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

"""
print(dataTrain.head())
print(dataTest.head())"""
"""print(dataTrain.info())
print(dataTest.info())"""

# String değeri int değere dönüştürdüm
dataTest["Sex"] = [1 if i.strip() =="male" else 0  for i in dataTest.Sex]
dataTrain["Sex"] = [1 if i.strip()  =="male" else 0  for i in dataTrain.Sex]

print(dataTrain.isna().sum())
print(dataTest.isna().sum())

print(dataTrain.describe().T)

#Boş verilerimizi doldurduk
dataTrain["Age"] =dataTrain["Age"].fillna(dataTrain["Age"].mean())
dataTest["Age"] = dataTest["Age"].fillna(dataTest["Age"].mean())

print(dataTrain.isna().sum())
print(dataTest.isna().sum())

#Detatlı inceleme yapacağız
"""corr_matrix=dataTrain.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.show()

sns.pairplot(dataTrain,diag_kind="kde",markers="+")
plt.show()
"""

#aykırı değerlerşi gönderme
"""for c in dataTrain.columns:#burda alt cizgi ve üst çizgi dışında kalanlar outliner
    plt.figure()
    sns.boxplot(x=c ,data=dataTrain,orient="v")
    plt.show()"""


x_train = dataTrain[["Pclass","Sex","Age"]]
y_train =dataTrain["Survived"]

x_test=dataTest[["Pclass","Sex","Age"]]

scaler=StandardScaler()#RobustScaler
x_train =scaler.fit_transform(x_train)
x_test =scaler.transform(dataTest)


model=RandomForestRegressor().fit(x_train,y_train)
y_pred =model.predict(x_test)
"""print(np.sqrt(mean_squared_error(y_train,y_pred)))
"""
for i in range(len(y_pred)):
    if y_pred[i]>=0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0


"""sns.countplot(y_pred)
plt.show()
"""


import csv

with open('ClassAgeSexPrediction.csv', mode='w') as csv_file:
    thewriter =csv.writer(csv_file)

    thewriter.writerow(["PassengerId","Survived"])
    xms = 891
    for i in range(418):

        xms =xms+1
        thewriter.writerow([xms,y_pred[i]])
# writing to csv file







