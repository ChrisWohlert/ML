#Titanic dataset predictions

#import panda library and a few others we will need.
from math import nan
from re import I
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
# skipping the header

data =pd.read_csv('D:/school/ML/Exercises/titanic_train_500_age_passengerclass.csv' , sep = ',' , header = 0 )

# show the data
print ( data .describe( include = 'all' ))
#the describe is a great way to get an overview of the data
print ( data .values)

# Replace unknown values. Unknown class set to 3
data["Pclass"].fillna(3, inplace = True)

# Replace unknown values. Unknown age set to 25

data.where(data['Pclass'] == '1', data['Age'], axis=1).fillna(2, inplace=True)
data.where(data['Pclass'] == '2', data['Age'], axis=1).fillna(2, inplace=True)
data.where(data['Pclass'] == '3', data['Age'], axis=1).fillna(2, inplace=True)
print(data)

yvalues = pd.DataFrame( dict ( Survived =[]), dtype = int )
yvalues[ "Survived" ] = data [ "Survived" ].copy()
#now the yvalues should contain just the survived column

x = data[ "Age" ]
y = data[ "Pclass" ]
plt.figure()
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
#plt.show()

#now we can delete the survived column from the data (because
#we have copied that already into the yvalues.
data.drop( 'Survived' , axis = 1 , inplace = True )

data.drop( 'PassengerId' , axis = 1 , inplace = True )

# show the data
print ( data.describe( include = 'all' ))

xtrain = data.head( 400 )
xtest = data.tail( 100 )

ytrain = yvalues.head( 400 )
ytest = yvalues.tail( 100 )

scaler = StandardScaler() 
scaler .fit(xtrain) 
xtrain = scaler .transform(xtrain) 
xtest= scaler .transform(xtest) 

mlp = MLPClassifier( hidden_layer_sizes =( 8 , 8 ), max_iter = 1000 , random_state = 0 ) 
mlp.fit(xtrain,ytrain.values.ravel()) 

predictions = mlp.predict(xtest) 

matrix = confusion_matrix(ytest,predictions) 
print (matrix) 
print('----------------------------------')
print (classification_report(ytest,predictions)) 

tn, fp, fn, tp = matrix.ravel() 
print(tn, fp, fn, tp)
print('Acc: ', (tp + tn) / (tn + fp + fn + tp))
