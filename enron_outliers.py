#!/usr/bin/python

import pickle
import pandas as pd
import sys
import matplotlib.pyplot
import numpy
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
#df = pd.DataFrame(data_dict)
#df.dropna(inplace = True)
del data_dict["TOTAL"]
#del data_dict["SKILLING JEFFREY K"]
#del data_dict["LAY KENNETH L"]
data = featureFormat( data_dict, features)
#data       = numpy.reshape( numpy.array(data), (len(data), 2))
#for salary in data_dict.items() :
#    print (salary)
#print(data_dict.i())

### your code below
#print(data_dict.keys())
for point in data:
    salary = point[0]
    bonus = point[1]
   # print(salary,bonus)

    matplotlib.pyplot.scatter( salary, bonus )

#from sklearn.linear_model import LinearRegression
#reg = LinearRegression()
#reg = reg.fit(salary,bonus)

for key, value in data_dict.items():
    if (str (value['bonus']) == 'NaN'):
    	value['bonus'] = 0

for key, value in data_dict.items():
    if (str (value['salary']) == 'NaN'):
    	value['salary'] = 0
    	

for key, value in data_dict.items():
    if (value['salary']>10**6 and value['bonus']>5*10**6):
    	print(key)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

