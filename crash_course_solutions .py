import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
np.random.seed(101)
array = np.random.randint(1,101,(100,5))
after_transformation = MinMaxScaler().fit_transform(array) 
array_in_pandas = pd.DataFrame(after_transformation)
array_in_pandas.columns = ['f1','f2','f3','f4','label']
X=array_in_pandas[0:len(array_in_pandas)-1]
#print(array_in_pandas[:-1])
Y=array_in_pandas[:-1]
train_x,train_y,test_x,test_y = train_test_split(X,Y,test_size=0.)
#plt.scatter(array_in_pandas[0],array_in_pandas[1])
#t=plt.imshow(array,aspect='auto')
#plt.title('My Plot')
#plt.colorbar(t)
print(train_x,train_y,test_x,test_y)
#plt.show()