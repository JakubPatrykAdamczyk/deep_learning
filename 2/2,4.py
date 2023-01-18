import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
# f(x)=y
# f(x+epsilon_x)=y+epsilon_y

# y_pred=dot(W,x)
# loss_value=loss(y_pred,y)

# #if x and y cons
# loss_value=f(W)
# # W=W0
# gradient(f)(W0)[i,j]
# W1=W0-step*gradient(f)(W0)


# x=np.array([1,2]) #wektor
# W=np.array([[1,2],[1,2]])#macierz
x=[]
y=[]
w=[]
for i in range(0,200):
    xn=i-100
    yn=2*xn**2+xn+1
    x.append(xn)
    y.append(yn)
w=np.array([x,y])
w=np.transpose(w)
print(w.shape)

network= models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(2,19902)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
w=to_categorical(w)
network.fit(w)
