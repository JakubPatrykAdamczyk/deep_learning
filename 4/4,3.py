#nadmierne, optymalne i słabe dopasowanie
from keras.datasets import imdb
import numpy as np
from keras import models,layers,optimizers,losses,metrics
import matplotlib.pyplot as plt
number_tent=10000

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=number_tent)
def vector_sequnece(sequnces,dimensions=number_tent):
    results=np.zeros((len(sequnces),dimensions))
    for i, sequnces in enumerate(sequnces):
        results[i,sequnces]=1
    return results

#towrzymy zbiory
x_train=vector_sequnece(train_data)
x_test=vector_sequnece(test_data)
#wektory etykiet próbek
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#optymalne
optima_model=models.Sequential()
optima_model.add(layers.Dense(16,activation='relu',input_shape=(number_tent,)))
optima_model.add(layers.Dense(16,activation='relu'))
optima_model.add(layers.Dense(1,activation='sigmoid'))

#słabe
small_model=models.Sequential()
small_model.add(layers.Dense(4,activation='relu',input_shape=(number_tent,)))
small_model.add(layers.Dense(4,activation='relu'))
small_model.add(layers.Dense(1,activation='sigmoid'))

#zbyt duże
big_model=models.Sequential()
big_model.add(layers.Dense(512,activation='relu',input_shape=(number_tent,)))
big_model.add(layers.Dense(512,activation='relu'))
big_model.add(layers.Dense(1,activation='sigmoid'))

def compile_and_fit(model):
    model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=4,batch_size=512)
    results=model.evaluate(x_test,y_test)
    print(results)

compile_and_fit(optima_model)
compile_and_fit(small_model)
compile_and_fit(big_model)
# Epoch 1/4
# 49/49 [==============================] - 9s 25ms/step - loss: 0.4822 - accuracy: 0.8088
# Epoch 2/4
# 49/49 [==============================] - 1s 20ms/step - loss: 0.2845 - accuracy: 0.9002
# Epoch 3/4
# 49/49 [==============================] - 1s 20ms/step - loss: 0.2224 - accuracy: 0.9204
# Epoch 4/4
# 49/49 [==============================] - 1s 19ms/step - loss: 0.1889 - accuracy: 0.9324
# 782/782 [==============================] - 3s 3ms/step - loss: 0.2872 - accuracy: 0.8838
# [0.28724104166030884, 0.8837599754333496]
# Epoch 1/4
# 49/49 [==============================] - 14s 26ms/step - loss: 0.6270 - accuracy: 0.6954
# Epoch 2/4
# 49/49 [==============================] - 1s 21ms/step - loss: 0.5025 - accuracy: 0.8549
# Epoch 3/4
# 49/49 [==============================] - 1s 19ms/step - loss: 0.3909 - accuracy: 0.8919
# Epoch 4/4
# 49/49 [==============================] - 1s 19ms/step - loss: 0.3080 - accuracy: 0.9112
# 782/782 [==============================] - 3s 3ms/step - loss: 0.3224 - accuracy: 0.8855
# [0.322417289018631, 0.8854799866676331]
# Epoch 1/4
# 49/49 [==============================] - 30s 234ms/step - loss: 0.4928 - accuracy: 0.7618
# Epoch 2/4
# 49/49 [==============================] - 8s 171ms/step - loss: 0.2669 - accuracy: 0.8926
# Epoch 3/4
# 49/49 [==============================] - 8s 171ms/step - loss: 0.2188 - accuracy: 0.9123
# Epoch 4/4
# 49/49 [==============================] - 8s 171ms/step - loss: 0.1628 - accuracy: 0.9375
# 782/782 [==============================] - 19s 9ms/step - loss: 0.6540 - accuracy: 0.7634
# [0.6540432572364807, 0.7633600234985352]