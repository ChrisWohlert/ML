import os
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier


def find_floor(x):
    if "Ground" in x:
        return 0
    if "Upper" in x:
        return -1
    if "Lower" in x:
        return -2
    return int(x.split()[0])

def find_floors(x): 
    x = x.split()
    if x[-1] == "Ground":
        return 0
    return int(x[-1])


print(os.getcwd())
dataset = pandas.read_csv(os.path.join(os.getcwd(), "Exercises/Kaggle/Housing/House_Rent_Dataset.csv"))

X = dataset.drop(["Rent"], axis=1)
y = dataset.take([2], axis=1)
print(X)

X["Posted On"] = X["Posted On"].apply(lambda x: (datetime.now() - datetime.strptime(x, "%Y-%m-%d")).days)
floor = X["Floor"].apply(find_floor)
floors = X["Floor"].apply(find_floors)
X["Floor"] = floor
X["Floors"] = floors
X["Area Type"] = X["Area Type"].factorize()[0]
X["Area Locality"] = X["Area Locality"].factorize()[0]
X["City"] = X["City"].factorize()[0]
X["Furnishing Status"] = X["Furnishing Status"].factorize()[0]
X["Tenant Preferred"] = X["Tenant Preferred"].factorize()[0]
X["Point of Contact"] = X["Point of Contact"].factorize()[0]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train)

#model = LinearRegression()
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(train_score)
print(test_score)
