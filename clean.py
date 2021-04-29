import pandas as pd
import numpy as np
from utils.utils import transf_duration
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from map import *

def clean_data(big_df):
    #Handling null values
    big_df['Total_Stops'] = big_df['Total_Stops'].fillna(value=big_df['Total_Stops'].mode()[0])
    big_df['Route'] = big_df['Route'].fillna(value=big_df['Route'].mode()[0])

    #Seperating Date_of_Journey column into date, month and year
    big_df['Date'] = big_df['Date_of_Journey'].str.split('/').str[0].astype(int)
    big_df['Month'] = big_df['Date_of_Journey'].str.split('/').str[1].astype(int)
    big_df['Year'] = big_df['Date_of_Journey'].str.split('/').str[2].astype(int)
    big_df.drop(['Date_of_Journey'],axis=1,inplace=True)

    #Extracting only the time part from Arrival_time feature
    big_df['Arrival_Time'] = big_df['Arrival_Time'].str.split(' ').str[0]

    #Seperating Arrival Time into Arrival Hour and Arrival minute
    big_df['Arrival_Hour'] = big_df['Arrival_Time'].str.split(':').str[0].astype(int)
    big_df['Arrival_Minute'] = big_df['Arrival_Time'].str.split(':').str[1].astype(int)

    #Seperating Departure time into Departure hour and Departure minute
    big_df['Departure_Hour'] = big_df['Dep_Time'].str.split(':').str[0].astype(int)
    big_df['Departure_Minute'] = big_df['Dep_Time'].str.split(':').str[1].astype(int)

    #Dropping the Arrival/Departure Time
    big_df.drop(['Arrival_Time','Dep_Time'],axis=1,inplace=True)

    #Handling route column
    big_df['Route_1'] = big_df['Route'].str.split('→ ').str[0]  # Seperating route parameter into different columns
    big_df['Route_2'] = big_df['Route'].str.split('→ ').str[1]
    big_df['Route_3'] = big_df['Route'].str.split('→ ').str[2]
    big_df['Route_4'] = big_df['Route'].str.split('→ ').str[3]
    big_df['Route_5'] = big_df['Route'].str.split('→ ').str[4]

    big_df['Route_1'].fillna("None", inplace=True)  # Handling null values in the newly created
    big_df['Route_2'].fillna("None", inplace=True)
    big_df['Route_3'].fillna("None", inplace=True)
    big_df['Route_4'].fillna("None", inplace=True)
    big_df['Route_5'].fillna("None", inplace=True)

    #Dropping the original Route column
    big_df.drop(['Route'],axis=1,inplace=True)

    #Converting Duration into minutes
    big_df['Duration'] = big_df['Duration'].apply(transf_duration)

    return big_df

def encode_categorical(cleaned_data):
    cleaned_data['Route_1'] = cleaned_data['Route_1'].str.strip()
    cleaned_data['Route_2'] = cleaned_data['Route_2'].str.strip()
    cleaned_data['Route_3'] = cleaned_data['Route_3'].str.strip()
    cleaned_data['Route_4'] = cleaned_data['Route_4'].str.strip()
    cleaned_data['Route_5'] = cleaned_data['Route_5'].str.strip()

    #Encoding categorical variables using label encoding because there is some rank associated with them
    cleaned_data['Airline'] = cleaned_data['Airline'].map(dic_airline)
    cleaned_data['Source'] = cleaned_data['Source'].map(dic_source)
    cleaned_data['Destination'] = cleaned_data['Destination'].map(dic_destination)
    cleaned_data['Total_Stops'] = cleaned_data['Total_Stops'].map(dic_totalstops)
    cleaned_data['Additional_Info'] = cleaned_data['Additional_Info'].map(dic_addinfo)
    cleaned_data['Route_1'] = cleaned_data['Route_1'].map(dic_route1)
    cleaned_data['Route_2'] = cleaned_data['Route_2'].map(dic_route2)
    cleaned_data['Route_3'] = cleaned_data['Route_3'].map(dic_route3)
    cleaned_data['Route_4'] = cleaned_data['Route_4'].map(dic_route4)
    cleaned_data['Route_5'] = cleaned_data['Route_5'].map(dic_route5)
    return cleaned_data

def feature_sel(encoded_data):
    var_thres = VarianceThreshold(threshold=0)
    var_thres.fit(encoded_data)
    constant_columns = [column for column in encoded_data.columns
                        if column not in encoded_data.columns[var_thres.get_support()]]
    encoded_data.drop(constant_columns, axis=1, inplace=True)
    return encoded_data