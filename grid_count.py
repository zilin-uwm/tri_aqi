

import pandas as pd
import numpy as np
import time
import datetime
from math import radians, cos, sin, asin, sqrt, pi
datapoi = pd.read_csv(r'/Users/zhangzilin/Desktop/paper/DataSet/available/useful/POI.csv')
datatra = pd.read_csv(r'/Users/zhangzilin/Desktop/paper/DataSet/available/useful/uber14.csv')
dataframe = pd.read_table(r"/Users/zhangzilin/Desktop/paper/DataSet/available/useful/tra.txt", delimiter=',')
#贾云开

#data = pd.read_csv(r'D:\finance\task3\task3\POI(2).csv')

#dataaqi = pd.read_csv(r'/Users/zhangzilin/Desktop/paper/DataSet/available/AQIdata.csv')


coord_range = np.array([[40.45325666, -74.27480126], [40.94787907, -73.70741171]])

lat_min = 40.45325666
lon_min = -74.27480126
lat = abs(40.45325666-40.94787907)
lon = abs(abs(-74.27480126)-abs(-73.70741171))
k = 10
lat_k = lat/k
lon_k = lon/k

list_1 = []
def grid_count(data_work):
    data_count = pd.DataFrame()
    for i in range(len(data_work)):
        Lat = data_work['Lat'][i]
        Lat_gap = abs(Lat-lat_min)
        Lat_int = int(Lat_gap/lat_k)
        Lon = data_work['Lon'][i]
        Lon_gap = abs(abs(Lon)-abs(lon_min))
        Lon_int = int(Lon_gap/lon_k)
        list_1.append([Lat_int,Lon_int])
        data_count= pd.DataFrame(list_1,columns =['Lat_count','Lon_count'])
        data_count['Lat_lon_count'] = "["+data_count['Lat_count'].apply(lambda x: str(x))+","+data_count['Lon_count'].apply(lambda x: str(x))+"]"

        #data_count_list = data_count['Lat_lon_count'].value_counts()
        #data_count_list = pd.DataFrame(data_count_list)
        #data_count_list.to_csv(r'/Users/zhangzilin/Desktop/paper/DataSet/result/POI_count_{}.csv'.format(k))
    return data_count



#count tra data by grid by date
data_tra = dataframe
data_tra['Date'] = dataframe['Date/Time'].apply(lambda x: x[0:10])
#datatra['Date/Time'] = pd.to_datetime(datatra['Date/Time'], format ='%Y-%m-%d')

#data_tra['Date/Time'] = data_tra['Date/Time'].apply(lambda x: datetime.datetime.date(x))
def get_one_day(data_tra,column_name):

    data_date = data_tra[str(column_name)].unique()

    data_date_df_best = pd.DataFrame()
    data_list = []
    for date_s in data_date[6:10]:

            data_oneday = data_tra[data_tra[str(column_name)] == date_s]
            data_oneday = data_oneday.reset_index()
            print(len(data_oneday))
            data_onedaynew  = grid_count(data_oneday)

            data_onedaynew_list = data_onedaynew['Lat_lon_count'].value_counts()
            data_onedaynew_list = pd.DataFrame(data_onedaynew_list).reset_index()
            data_onedaynew_list['date'] = date_s
            data_date_df_best = pd.concat([data_date_df_best,data_onedaynew_list])
    print (data_date_df_best)
    return data_date_df_best

data_tra_one_day = get_one_day(data_tra,'Date')
data_tra_one_day.to_csv(r'/Users/zhangzilin/Desktop/paper/DataSet/result/tra6.csv')


#使用经纬度计算两点间距离（m）
from math import radians, cos, sin, asin, sqrt

lon1, lat1, lon2, lat2 = (-74.27480126, 40.45325666, -73.70741171, 40.94787907)

def geodistance(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)]) # 经纬度转换成弧度
    lon=lon2-lon1
    lat=lat2-lat1
    a=sin(lat/2)**2 + cos(lat1) * cos(lat2) * sin(lon/2)**2
    dist=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    dist=round(dist/1000,3)
    return dist


#类似DBSCAN的周围点计数

datatra4 = pd.read_csv(r'/Users/zhangzilin/Desktop/paper/DataSet/available/useful/traDATA4.csv')
datatra4 = pd.DataFrame(datatra4)
data_date = datatra4['date'].unique()

for data_str in data_date:
    data_min = datatra4[datatra4['date']==data_str].reset_index()
    #data_index =data_min['index'].unique()
    for index_i in data_min['index']:
        num = 0
        list_neigh= []
        x = int(index_i[1])
       # print x
        y = int(index_i[3])
        #print y
        for i in range(x-1,x+2):
            for j in range(y-1,y+2):

                result = data_min[data_min['x'] == i and data_min['y'] == j].reset_index
                print result
                #str_check_index = '[{},{}]'.format(i,j)
                #print str_check_index

                if result['index'] in data_min['index']:
                    #data_log = data_min[data_min['index'] == '[{},{}]'.format(i,j)].reset_index
                    #print data_log

                    num =+ result[0 ,'Lat_lon_count']
            print(num)
        #print("{}".format(num))
    list_neigh.append([index_i,num])

