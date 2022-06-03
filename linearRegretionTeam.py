import sklearn # meka udauwenne model eka hadanna
import matplotlib.pyplot as plt # this is use to drow the prathara
import numpy as np
from sklearn import model_selection
from sklearn import linear_model

#y =mx + c akaraye
# F = 1.8*c + 32 # selsiyas FerendHight karana akaraya balanna yanne

x = list(range(0, 50)) # celsiyas anshaka agayan, independ vereable
y = [1.8*F +32 for F in x] # ferendHight dependen vereable
print(f'X: {x}')
print(f'Y: {y}')

plt.plot(x,y,'-*r') # '-*r' walin kiyanne prasthare pata   prastharaya adinne methena
plt.show()

# sema data ekakma array ekakata dala wena wenama ganna eka karanne, mehi reshape(-1,1) kiyanne eka eliment ekaka pamanak ganna kiyana eka
# sklearn lib eka bavitha karannam data e vidiyata denna ona
X = np.array(x).reshape(-1, 1)
Y = np.array(y).reshape(-1, 1)

# data set eka kadaganna ona testing walatai trening walatai
# mema function eken puluwan apita awashya testing presentage labadi deta set eka kadaganna
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=0.2)
# model eka create kirema
model = linear_model.LinearRegression()
# data set eka model ekat pass kirima
model.fit(xTrain, yTrain)
print(model.coef_) # model eke anucramanaya genima
print(model.intercept_) # model eke anthakkandaya genima


# apee model eke accuracy eka belima
accuracy = model.score(xTest, yTest)
print(accuracy*100)




































