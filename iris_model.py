#Importaciones
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle

#Carga de datos
iris = datasets.load_iris()

X = iris.data
Y = iris.target

#Separar los datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_m = SVC()

#Entrenamiento de modelos
lin_regr = lin_reg.fit(X_train,Y_train)
log_regr = log_reg.fit(X_train,Y_train)
svc_mo = svc_m.fit(X_train, Y_train)

with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)
    
with open('log_reg.pkl', "wb") as lo:
    pickle.dump(log_reg, lo)
    
with open('svc_m.pkl', 'wb') as sv:
    pickle.dump(svc_m, sv)