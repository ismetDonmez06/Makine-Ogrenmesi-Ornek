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
from sklearn import model_selection
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


column_name =["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]
data =pd.read_csv("auto-mpg.data", names=column_name, na_values="?", comment="\t", sep=" ", skipinitialspace=True)

data =data.rename(columns ={"MPG":"target"})

#Verimizi Yüzeysel inceliyoruz
print(data.head())
print(data.shape)
print(data.info()) # horsepowerde 6 tane boşluk var
print(data.describe().T)
print()

#Boş value değerlerini halletcez Missing Value
print(data.isna().sum()) #boş verilerin toplamını görüyoruz
data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())#bOŞ değerler ortalamasını attık
print(data.isna().sum())
sns.displot(data.Horsepower)


#Keşifsel Veri analizi (Detaylı veri analiz yapacağız)


corr_matrix =data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.show()

threshold = 0.75
filtre = np.abs(corr_matrix["target"])>threshold
corr_feature =corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_feature].corr(),annot=True,fmt=".2f")
plt.show()


"""
multicollinearity 
"""

"""sns.pairplot(data,diag_kind="kde",markers="+")
plt.show()
"""
"""
cylinder and origin can be categorical (feature enginering yapcaz burda)
"""

#box

for c in data.columns:#burda alt cizgi ve üst çizgi dışında kalanlar outliner
    plt.figure()
    sns.boxplot(x=c ,data=data,orient="v")
    plt.show()

#outliner Tespiti ve çıkarılması
# horsepower and acceleration

"""thr =2
horsepower_describe=describe["Horsepower"]
print(horsepower_describe)
"""


#Feature Enginering (normalleştirme)
#skewness değerlerini normalleştirme skew >1  sağa doğru kuyruk olur
#skew <-1 sola doğru kuyruk olur
#bunları cıkarcaz çünkü aşırı değerlerdir

# target depented variable

skewed_fats =data.apply(lambda x : skew(x.dropna())).sort_values(ascending=False)
skewness=pd.DataFrame(skewed_fats,columns=["skewed"])
print(skewness) #1 den büyükse poz skewness küçükse neg skewness var
#skewness lık olmadığı için burda işlem yapmıcaz eğer olsaydı Box Cox Transformatipn yontemiyle  halledecektik

#one hot encoding (Categorical verileri encoding yaparız çünkü veriyi bozar)
#bilgisayarın anlıcağı şekle getirmek 3 = 0 1 1 gibi

data["Cylinders"] =data["Cylinders"].astype(str)
data["Origin"] =data["Origin"].astype(str)

data =pd.get_dummies(data)

#split ve standartlaştırma
x=data.drop(["target"],axis=1)
y=data.target

X_train,X_test ,Y_train,Y_test =train_test_split(x,y,test_size=0.9,random_state=42)

#standarlaştırma
scaler=StandardScaler()#RobustScaler
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)


# lineer Regrosyonla veri eğitimi
lr = LinearRegression().fit(X_train,Y_train)
y_pred =lr.predict(X_test)
mse =mean_squared_error(Y_test,y_pred)
print("Lineer Regresyon MSE :" ,mse)











