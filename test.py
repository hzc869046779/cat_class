import pandas as pd


path = 'E:/研究生学习文件/研究生相关信息/大气预测/大气预测相关数据/Dump20190325/air_bj_airquality_station.sql'
data = pd.read_sql(path)
t = data.describe()