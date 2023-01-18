import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt



#dane boston
from keras.datasets import boston_housing

number_tent=10000
(train_data,train_labels),(test_data,test_labels)=boston_housing.load_data()

#normalizowanie danych
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

#funkcja do budowy modelu
def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                loss='mse',#średnia błędu kwadratowego
                metrics=['mae'])#średni błąd bezwzględny
    return model

#walidacja k składników (t-trening,w-walidacja)
#[w][t][t][t]
#[t][w][t][t]
#[t][t][w][t]
#[t][t][t][w]
k=4
num_val_sample=len(train_data)//k
num_epochs=100
all_scores=[]
for i in range(k):
    # print('procesing fold#',i)
    val_data=train_data[i*num_val_sample:(i+1)*num_val_sample]
    val_labels=train_labels[i*num_val_sample:(i+1)*num_val_sample]

    partial_train_data=np.concatenate(
        [
            train_data[:i*num_val_sample],
            train_data[(i+1)*num_val_sample:]
        ],
        axis=0
    )

    partial_train_labels=np.concatenate(
        [
            train_labels[:i*num_val_sample],
            train_labels[(i+1)*num_val_sample:]
        ],
        axis=0
    )
    model=build_model()
    history=model.fit(partial_train_data,
                        partial_train_labels,
                        validation_data=(val_data,val_labels),
                        epochs=num_epochs,
                        batch_size=1,
                        verbose=0)
    # mae_history=history['val_mean_absolute_error']
    
    mae_history=history['val_mae']
    all_scores.append(mae_history)
    #średnia
    avarge_mae_history=[np.mean([x[i]for x in all_scores])for i in range(num_epochs)]

plt.plot(range(1,len(avarge_mae_history)+1,avarge_mae_history))
plt.xlabel('licznik epok')
plt.ylabel('sredni blad bezwzgledny')
plt.show()
