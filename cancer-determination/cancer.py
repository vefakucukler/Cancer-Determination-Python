import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster.k_means_ import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import pandas as pd    ##gerekli kütüphaneleri ekleme (?)
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
cancer=pd.read_csv('cancer.data') ##cancer veri setini yükleme işlemi
cancer.head() ##cancer setinin başlığını (sütunların isimlerini) gösteriyor
cancer.shape ## (satır,sütun) sayısını gösteriyor
cancer=cancer.drop(['id','Unnamed: 32'],axis=1) ## id Sütununu siliyor.
cancer['diagnosis'].unique()
cancer.describe()

def encoding(cancer):
    le = LabelEncoder()
    for col in cancer.columns:
        if cancer[col].dtypes == 'object':
            cancer[col]=le.fit_transform(cancer[col])
            return cancer

def splitting (X,Y):
    return train_test_split(X,Y,test_size = 0.30,random_state = 42)

encoded_cancer = encoding(cancer) ## M ve B değerlerini 1 ve 0 yaptı (?)
encoded_cancer.head()

Y = cancer.diagnosis ## Y'nin içinde yalnızca diagnosis var şuan
X = cancer.drop('diagnosis',axis = 1) ##X in içinde cancerin diagnosis sütununsuz hali var şuan

x_train,x_test,y_train,y_test = splitting(X,Y)

lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
accl = accuracy_score(y_test,y_pred)

matl = confusion_matrix(y_pred,y_test)

################### using...
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(chi2,20).fit_transform(X,Y)
print(X_new.shape) ##çıktı

x_train,x_test,y_train,y_test = splitting(X_new,Y)
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
accl = accuracy_score(y_test,y_pred)
print(accl)

n = len(Y)

X1 = np.ones((n,1))
X_transformed = pd.DataFrame(np.hstack((X1,X)))
x_train,x_test,y_train,y_test = splitting(X_transformed,Y)

def linear_regression(x_train,y_train,x_test):
    x_trans = np.transpose(x_train)
    product = np.dot(x_trans,x_train)
    inverse = np.linalg.inv(product)
    product = np.dot(inverse,x_trans)
    result = np.dot(product,y_train)
    y_pred2 = np.dot(x_test,result)
    return y_pred2

def sigmoid(data):
    for i in range(len(data)):
        data[i] = 1/(1+np.exp(-data[i]))
        return data
    
y_pred2 = linear_regression(x_train,y_train,x_test)
y_pred3 = sigmoid(y_pred2)

for i in range(len(y_pred3)): ##
    if y_pred3[i]>0.5:
        y_pred3[i] = 1
    else :
        y_pred3[i] = 0

acc2 = accuracy_score(y_test,y_pred3)
acc2

models = [
    
    ('KNN', KNeighborsClassifier()),
  
]

# Modeller için 'cross validation' sonuçlarının  yazdırılması
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
   
    
print('***********KNN**************')  
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print('***********PCA**************')

pca = PCA()
X_train1 = pca.fit_transform(x_train)
X_validation1 = pca.transform(x_test)
#print(pca.explained_variance_ratio_)  #her bileşen için varyans değerlerni gösteriyor
print('************************')
print("\n\n",X_train1)
print('*******KMEANS*************')

kmeans = KMeans(n_clusters=5)
kmeans.fit(X,Y)
#print(kmeans.cluster_centers_)
print(pd.crosstab(Y,kmeans.labels_))


