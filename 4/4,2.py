#k krotna walidacja krzyżowa

from keras.datasets import imdb
import numpy as np
from keras import models,layers,optimizers,losses,metrics
import matplotlib.pyplot as plt

data=[]#dane wejściowe 
np.random.shuffle(data)
k=4
num_val_sample=len(data)//k

num_epochs=100
all_scores=[]
for i in range(k):
    # print('procesing fold#',i)
    val_data=data[i*num_val_sample:(i+1)*num_val_sample]
    val_labels=data[i*num_val_sample:(i+1)*num_val_sample]

    partial_train_data=np.concatenate(
        [
            data[:i*num_val_sample],
            data[(i+1)*num_val_sample:]
        ],
        axis=0
    )

    partial_train_labels=np.concatenate(
        [
            data[:i*num_val_sample],
            data[(i+1)*num_val_sample:]
        ],
        axis=0
    )
    model=build_model()# funkcja z 3,4
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