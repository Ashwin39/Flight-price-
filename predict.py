import pickle
from train import modelfilename,filename1
from map import *
model = pickle.load(open(modelfilename,'rb'))
scmodel = pickle.load(open(filename1,'rb'))

def predict(predf):
    predf['Airline'] = predf['Airline'].map(dic_airline)
    predf['Source'] = predf['Source'].map(dic_source)
    predf['Destination'] = predf['Destination'].map(dic_destination)
    predf['Total_Stops'] = predf['Total_Stops'].map(dic_totalstops)
    predf['Additional_Info'] = predf['Additional_Info'].map(dic_addinfo)
    predf['Route_1'] = predf['Route_1'].map(dic_route1)
    predf['Route_2'] = predf['Route_2'].map(dic_route2)
    predf['Route_3'] = predf['Route_3'].map(dic_route3)
    predf['Route_4'] = predf['Route_4'].map(dic_route4)
    predf['Route_5'] = predf['Route_5'].map(dic_route5)

    transform = scmodel.transform(predf)
    predictresult = model.predict(transform)
    return predictresult