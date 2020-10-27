import datasets

X,Y = datasets.load_linear_example1()

print(X)
print(X[0])
print(Y)

import regression
model = regression.LinearRegression()
print(model.x)

#test fit
import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X,Y)
print(model.theta)

#test predict
model.predict(X)
print(model.predict(X))

#test score
model.score(X,Y)
print(model.score(X,Y))