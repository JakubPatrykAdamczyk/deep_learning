from keras.datasets import imdb
import numpy as np
from keras import models,layers,optimizers,losses,metrics
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

#definicja modelu
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(number_tent,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#dawna kompilacja
# network.compile(optimizer='rmsprop',
#     loss='categorical_crossentropy',
#     metrics=['accuracy'])

#korzystając z funkcji
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy])

#tworzenie zbioru walidacyjnego
x_val=x_train[:number_tent]
partial_x_train=x_train[number_tent:]
y_val=y_train[:number_tent]
partial_y_train=y_train[number_tent:]




#trenowanie modelu
history=model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))

history_dict=history.history
print(history_dict.keys())
