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
network.add(layers.Dense(512,activation='relu',input_dim=1))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',
    loss='mse',
    metrics=['accuracy'])
# w=to_categorical(w)
network.fit(w[:,0],w[:,1],epochs=100,verbose=0)

#rebuild
model=models.Sequential()
model.add(layers.Dense(10,input_dim=1))
model.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['acc'])

# w=to_categorical(w)
model.fit(w[:,0],w[:,1],epochs=100,verbose=0)#verbose = 0 silent
trash,results=model.evaluate(w[:,0],w[:,1])
print("acc100:",results) #0.57

# model.fit(w[:,0],w[:,1],epochs=1000,verbose=0)#verbose = 0 silent
# trash,results=model.evaluate(w[:,0],w[:,1])
# print("acc1000:",results) #0.57

#ok change y
#clear all
x,y,w=[],[],[]
for i in range(200):
    x.append((i-100))
    y.append((x[i]**2))
   
w=np.array([x,y])
w=np.transpose(w)
model.fit(w[:,0],w[:,1],epochs=100,verbose=0)#verbose = 0 silent
trash,results=model.evaluate(w[:,0],w[:,1])
print("acc100:",results) #0.57

#ok change x to start from 0
x,y,w=[],[],[]
for i in range(200):
    x.append(i)
    y.append((2*x[i]**2+x[i]+1))
   
w=np.array([x,y])
w=np.transpose(w)
model.fit(w[:,0],w[:,1],epochs=100,verbose=0)#verbose = 0 silent
trash,results=model.evaluate(w[:,0],w[:,1])
print("acc100:",results) #1

#and little random
w=np.random.rand(200,2)
# print(w)
w=np.array([x,y])
w=np.transpose(w)
model.fit(w[:,0],w[:,1],epochs=100,verbose=0)#verbose = 0 silent
trash,results=model.evaluate(w[:,0],w[:,1])
print("accrand:",results) #1
#why?
model2=models.Sequential()
model2.add(layers.Dense(10,input_dim=1))

model2.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['acc'])
model2.fit(w[:,0],w[:,1],epochs=100,verbose=0)#verbose = 0 silent
trash,results=model2.evaluate(w[:,0],w[:,1])
print("accrand:",results) #1

