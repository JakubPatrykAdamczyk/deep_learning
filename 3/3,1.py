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
#wykres strat trenowania i walidacji
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label='Strata trenowania')
plt.plot(epochs,val_loss,'b',label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('epoki')
plt.ylabel('strata')
plt.show()

#wykres dokładności
plt.clf()#clear
acc_value=history_dict['acc']
val_acc_values=history_dict['val_acc']
plt.plot(epochs,acc,'bo',label='Dokladnosc trenowania')
plt.plot(epochs,val_acc,'b',label='Dokladnosc walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('epoki')
plt.ylabel('strata')
plt.show()

#ponowne trenowanie modelu
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(number_tent,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4,batch_size=512)
results=model.evaluate(x_test,y_test)
print(results)



