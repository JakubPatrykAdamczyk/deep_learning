#prosta walidacja na odłożonych danych
from keras.datasets import imdb
import numpy as np
from keras import models,layers,optimizers,losses,metrics
import matplotlib.pyplot as plt
number_tent=10000

#losowe dane

data=np.arange((number_tent*number_tent)).reshape((number_tent,number_tent))
np.random.shuffle(data)

#zbiór walidacyjny
validation_data=data[:number_tent]
data=data[number_tent:]

#zbiór treningowy
train_data=data[:]

#trening
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop',
            loss='mse',#średnia błędu kwadratowego
            metrics=['mae'])#średni błąd bezwzględny

model.train(train_data)
print(model.evaluate(validation_data))


