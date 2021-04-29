import pandas as pd
import numpy as np
from clean import clean_data,encode_categorical,feature_sel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

#Read the excel file
big_df = pd.read_excel('./Data/Data_Train.xlsx')
modelfilename = 'flightpredmodel.sav'
filename1 = 'standardization.pickle'

def train():

    cleaned_data = clean_data(big_df)
    encoded_data = encode_categorical(cleaned_data)
    featured_data = feature_sel(encoded_data)

    X = featured_data.drop(['Price'],axis=1)
    y = featured_data.Price

    #Splitting and Standardizing dataset
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pickle.dump(sc,open(filename1,'wb'))

    #Build model using XGBoost
    xgmodel = XGBRegressor()
    forestmodel = RandomForestRegressor()
    param_grid = {

        'learning_rate': [1, 0.5, 0.1, 0.01, 0.001],
        'max_depth': [3, 5, 7, 10, 20],
        'n_estimators': [10, 50, 100, 200]

    }

    param_gridf = {
        'n_estimators': [100, 200, 300, 1000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [80, 90, 100, 110],
        'min_samples_split': [2, 5, 10, 15, 100],
        'min_samples_leaf': [1, 2, 5, 10]

    }
    grid= GridSearchCV(xgmodel,param_grid, verbose=3,cv=2)
    grid.fit(X_train,y_train)

    gridf = GridSearchCV(forestmodel,param_gridf, verbose=3)
    gridf.fit(X_train,y_train)


    xgmodel = XGBRegressor(**grid.best_params_)
    xgmodel.fit(X_train,y_train)
    y_pred = xgmodel.predict(X_test)
    xgbscore = r2_score(y_test,y_pred)
    print(xgbscore)


    forestmodel = RandomForestRegressor(**gridf.best_params_)
    forestmodel.fit(X_train,y_train)
    y_predf = forestmodel.predict((X_test))
    forestscore = r2_score(y_test,y_predf)
    print(forestscore)

    if(xgbscore>forestscore):
        pickle.dump(xgmodel, open(modelfilename, 'wb'))
    else:
        pickle.dump(forestmodel, open(modelfilename, 'wb'))

if __name__ == '__main__':
    train()