# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:49:37 2023

@author: Alexandra
"""

#importam librariile cele mai importante 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statistics
#importam setul de date
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
dfRealEstate=pd.read_excel(url)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

#ștergem coloana No din setul de date, deoarece indexul este dat automat de anaconda 
dfRealEstate.drop(["No"], axis = 1, inplace = True)

#redenumim coloanele
dfRealEstate=dfRealEstate.rename(columns = {'X1 transaction date':'TrDate',
                                            'X2 house age':'HouseAge',
                                            'X3 distance to the nearest MRT station':'DistMRT',                      'X4 number of convenience stores':'NrStores',
                                            'X5 latitude':'lat','X6 longitude':'long','Y house price of unit area':'HousePrice'})

#PRELUCRARE COLOANĂ "TrDate"
#dfRealEstate.TrDate.dtype = float64
def TrDate(d):
    if d["TrDate"] == 2013.4166667:
        return "2013"
    elif d["TrDate"] == 2013.500000:
        return "2013"
    elif d["TrDate"] == 2013.0833333:
        return "2013"
    elif d["TrDate"] == 2012.9166667:
        return "2012"
    elif d["TrDate"] == 2013.250000:
        return "2013"
    elif d["TrDate"] == 2012.8333333:
        return "2012"
    elif d["TrDate"] == 2012.6666667:
        return "2012"
    elif d["TrDate"] == 2013.3333333:
        return "2013"
    elif d["TrDate"] == 2013.000000:
        return "2013"
    elif d["TrDate"] == 2012.750000:
        return "2012"
    elif d["TrDate"] == 2013.1666667:
        return "2013"
    elif d["TrDate"] == 2013.5833333:
        return "2013"
    else:
        return "-"
    
#creare coloana TrYear care conține doar anii în care s-au făcut tranzacții
#această coloană este adăgată la sfârșitul dataframe-ului(setului de date)
dfRealEstate["TrYear"] = dfRealEstate.apply(TrDate, axis=1)

#dorim să inserăm coloana imediat după coloana "TrDate"
#duplicăm coloana "TrYear"
dfRealEstate.insert(1, "TrYear", dfRealEstate["TrYear"], allow_duplicates= True) 

#ștergem coloana duplicată de la finalul dataframe-ului 
dfRealEstate = dfRealEstate.iloc[:, :-1]

#CORELAȚII
corelatii=dfRealEstate.corr()
print(corelatii)
hmap = sns.heatmap(corelatii,cmap = "RdPu", annot=True, linewidth = .5)


#Corelația între HousePrice și DistrMRT 
data = pd.DataFrame(dfRealEstate)
sns.regplot(x='DistMRT', y='HousePrice', data=dfRealEstate)
plt.show()

#Corelația între HousePrice și NrStores
data = pd.DataFrame(dfRealEstate)
sns.regplot(x='NrStores', y='HousePrice', data=dfRealEstate)
plt.show()

#Corelația între DistMRT și NrStores
data = pd.DataFrame(dfRealEstate)
sns.regplot(x='NrStores', y='DistMRT', data=dfRealEstate)
plt.show()

#Grafice care arată anul tranzacției și prețul caselor 
data = pd.DataFrame(dfRealEstate) 
sns.barplot(x='TrYear', y='HousePrice', data=dfRealEstate)
plt.show()

data = pd.DataFrame(dfRealEstate)
sns.stripplot(x='TrYear', y='HousePrice', data=dfRealEstate)
plt.show()








