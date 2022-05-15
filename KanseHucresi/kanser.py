
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
filterwarnings("ignore")


data =pd.read_csv("data.csv")

data =data.rename(columns={'diagnosis':'target'})

# String veriyi int çeviridk
data["target"] = [1 if i.strip() =="M" else 0  for i in data.target]



"""
standardization-normalize
missing value = none
"""

#korülasyon yapalım (veriler arasında ilişkiyi inceliyoruz)
corr_matrix =data.corr()
sns.clustermap(corr_matrix,annot =True,fmt=".2f")
plt.show()

#birbirleriyle yüksek işilkili verileri daha rahat bulma 1 en iyi -1 en kötü ilişiki

theshold =0.75
filtre =np.abs(corr_matrix["target"])>theshold
corr_features =corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot =True,fmt=".2f")
plt.show()


#pair plot en etkili görselleştirme
sns.pairplot(data[corr_features],diag_kind="kde",markers="+",hue="target")
plt.show()


#outlier değerleri yani aykırı değerleri düzeltme (Density based ODS (LOCAL OUTLİNER FACTOR(yontemi kullancaz)))

y=data.target
x =data.drop(["target"],axis=1)
columns =x.columns.tolist()

clf=LocalOutlierFactor()
y_pred=clf.fit_predict(x)
X_score =clf.negative_outlier_factor_

outliner_score =pd.DataFrame()
outliner_score["score"]=X_score #bu işlemi yaptıktan sonra aykırı degerler hakkında yorum yapcaz

print(outliner_score["score"])

#threshold
thesholdum =-2.5
filtre =outliner_score["score"]<thesholdum
outliner_index =outliner_score[filtre].index.tolist()
plt.figure()
plt.scatter(x.iloc[outliner_index,0],x.iloc[outliner_index,1],color="blue",s=50,label="Outliner")

#gorselleştirelim
plt.scatter(x.iloc[:,0],x.iloc[:,1],color="k",s=3,label="Data points")
radius = (X_score.max()-X_score)/(X_score.max()-X_score.min())
outliner_score["radius"]=radius
plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r",facecolors="none")
plt.legend()
plt.show()


# outliner silme
x=x.drop(outliner_index)
y=y.drop(outliner_index).values


#verimizi bölelim

X_train,X_test ,Y_train,Y_test =train_test_split(x,y,test_size=0.3,random_state=42)


#standadization (normalization) verinin yeniden ölçeklendirilmesi
#degerler arasında çok fark varsa yapılır genelde yapılır

scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)


# verimizi eğitecez

knn =KNeighborsClassifier(n_neighbors=2).fit(X_train,Y_train)
y_pred =knn.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
acc=accuracy_score(Y_test,y_pred)
score = knn.score(X_test,Y_test)
print("Score :", score)
print("cm",cm)
print("Basic :" ,acc)


# en iyi paremetreyi seçme

def KNN_Best_Params(x_train, x_test, y_train, y_test):
    k_range = list(range(1, 45))
    weight_options = ["uniform", "distance"]
    print()
    param_grid = dict(n_neighbors=k_range, weights=weight_options)

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
    grid.fit(x_train, y_train)

    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()

    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)

    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)

    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ", cm_test)
    print("CM Train: ", cm_train)

    return grid


grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)

## train ve test sonucları arasındaki fark %4ü geçmiyecek ve başarı oranımız %90 üzerinde olursa
## başarılıyız dır eğer aradaki fark çoksa overfitting olur eğer eğere başarımız
# test %70  train %72 olsaydı burdada underfitting yani az öğrenme olmuş olacaktır
# test %90  train %97 olsaydı burdada overfitting yani aşırı öğrenme olmuş olacaktır

