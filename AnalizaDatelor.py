# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:08:18 2023

@author: Adelina
"""
#importam librariile cele mai importante 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statistics
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


#importam setul de date
dfRealEstate=pd.read_excel("Real estate valuation data set.xlsx")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)



#redenumesc coloanele pentru comoditate
dfRealEstate=dfRealEstate.rename(columns = {'No':'No','X1 transaction date':'TrDate','X2 house age':'HouseAge',
                               'X3 distance to the nearest MRT station':'DistMRT','X4 number of convenience stores':'NrStores',
                               'X5 latitude':'lat','X6 longitude':'long','Y house price of unit area':'HousePrice'})
 

#afisarea primelor 5 inregistrari
print(dfRealEstate.head())

#se verifica daca exista valori nule in setul de date
print(dfRealEstate.isnull().sum())
dfRealEstate.info()

#medii, cvartile,deviatie standard
dfRealEstate.describe()
print("mediana pentru Transaction date",statistics.median(dfRealEstate["TrDate"]))
print(statistics.median(dfRealEstate["HouseAge"]))
print(statistics.median(dfRealEstate["DistMRT"]))
print(statistics.median(dfRealEstate["NrStores"]))
print(statistics.median(dfRealEstate["lat"]))
print(statistics.median(dfRealEstate["long"]))
print(statistics.median(dfRealEstate["HousePrice"]))


#multicoliniaritatea

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_variables=dfRealEstate[["TrDate","HouseAge","DistMRT","NrStores","lat","long"]]
vif_data=pd.DataFrame()

vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]

print(vif_data)

#regresii LinearRegression

#regresorii
X=dfRealEstate[["HouseAge","DistMRT","NrStores"]]

#variabila dependenta
y=dfRealEstate['HousePrice']


#impartirea setului de date in train si test
X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=0)


#potrivirea modelului de regresie pe setul de date de training
lm=LinearRegression()
model=lm.fit(X_train, y_train)

print("constanta=", model.intercept_)
print("lista coeficientilor beta=", model.coef_)


#se fac previziuni bazate pe setul de date de testare
#si se testeaza prezicerile pe setul de date de testare
predictions=lm.predict(X_test)

sns.regplot(x=predictions, y=y_test)

plt.scatter(y_test, predictions)
plt.xlabel("Date reale")
plt.ylabel("Date prezise")
plt.show()


#coeficientul de determinare r2

from sklearn.metrics import r2_score
r2 = r2_score(y_test,predictions)
print("r2=",r2)

#cross-validation
from sklearn.model_selection import cross_val_score, cross_val_predict

scores=cross_val_score(model, X_train,y_train ,cv=6)
print ("scores=",scores)
print("mean score=",scores.mean())
print("dev stnd=", scores.std())


from sklearn.metrics import r2_score
model_p=LinearRegression()
cv_predict = cross_val_predict(estimator=model_p,X=X_train,y=y_train,cv=6)
#print("predicted scores", cv_predict)
r2_cv_predict= r2_score(y_true=y_train, y_pred = cv_predict)
print("cross validated r2=",r2_cv_predict)

#Ridge regression
model_Ridge=Ridge(alpha=1)
model_Ridge.fit(X_train,y_train)

y_prezis_Ridge= model_Ridge.predict(X_test)
print("r^2_alpha_Ridge=1:",r2_score(y_test,y_prezis_Ridge))


model_Ridge=Ridge(alpha=50)
model_Ridge.fit(X_train,y_train)

y_prezis_Ridge= model_Ridge.predict(X_test)
print("r^2_alpha_Ridge=50:",r2_score(y_test,y_prezis_Ridge))

#lasso 
model_Lasso=Lasso(alpha=1000)
model_Lasso.fit(X_train, y_train)

y_prezis_Lasso=model_Lasso.predict(X_test)
print("r^2_alpha_Lasso=1000 :", r2_score(y_test,y_prezis_Lasso))

print(model_Lasso.coef_)


#elastic net
model_EN=ElasticNet(alpha=0.1,l1_ratio=0.25)
model_EN.fit(X_train,y_train)

y_prezis_EN=model_EN.predict(X_test)
print("r^2_alpha_EN=1 :", r2_score(y_test,y_prezis_EN))
