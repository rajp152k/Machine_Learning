import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#creating the linear regressor
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#creating the polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
polyfeats = PolynomialFeatures(degree = 4)

X_Poly = polyfeats.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_Poly,y)

#visualizing the results

plt.scatter(X,y,color = 'blue')
plt.plot(X,lin_reg.predict(X),color = 'red')
plt.title('BLUFF???')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()

figure,
plt.scatter(X,y,color = 'blue')
plt.plot(X,lin_reg_2.predict(polyfeats.fit_transform(X)),color= 'red')
plt.title('BLUFF???')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()
      
      lin_reg_2.predict(polyfeats.fit_transform(6.5))