import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt



#dane z reuters
from keras.datasets import reuters

number_tent=10000
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=number_tent)

#dekodowanie indeksów
word_index=reuters.get_word_index()
reverse_word_index=dict([(value,key)for(key,value)in word_index.items()])

decode_newswire=''.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])

def vectorize_sequences(sequences,dimension=number_tent):
    results=np.zeros((len(sequences),dimension))
    for i, sequences in enumerate(sequences):
        results[i,sequences]=1
    return results

#towrzymy zbiory
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)
#wektory etykiet próbek
# y_train=np.asarray(train_labels).astype('float32')
# y_test=np.asarray(test_labels).astype('float32')

#tworzymy funkcję zastępującą to_categorical z keras
def to_one_hot(labels,dimensions=46):
        results=np.zeros((len(labels),dimensions))
        for i, labels in enumerate(labels):
            results[i,labels]=1
        return results

one_hot_train_label=to_one_hot(train_labels)
one_hot_test_label=to_one_hot(test_labels)

#definicja modelu
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(number_tent,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

#kompilacja
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#tworzenie zbioru walidacyjnego
one_tau=1000
x_val=x_train[:one_tau]
partial_x_train=x_test[one_tau:]

y_val=one_hot_train_label[:one_tau]
partial_y_train=one_hot_test_label[one_tau:]

#trenowanie
history=model.fit(partial_x_train,
                partial_y_train,
                epochs=20,
                batch_size=512,
                validation_data=(x_val,y_val),
                verbose=0)
trash,results=model.evaluate(partial_x_train,partial_y_train)
print("accrand:",results) #0.95

# #wykres

# loss=history.history['loss']
# val_loss=history.history['val_loss']
# epochs=range(1,len(loss)+1)

# plt.plot(epochs,loss,'bo',label='Strata trenowania')
# plt.plot(epochs,val_loss,'b',label='Strata walidacji')
# plt.title('Strata trenowania i walidacji')
# plt.xlabel('epoki')
# plt.ylabel('strata')
# plt.show()   

#generowanie przewidywań
prediction=model.predict(x_test)
#print(prediction.shape)

#model zbyt wąski
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(number_tent,)))
model.add(layers.Dense(4,activation='relu'))#dawniej 64
model.add(layers.Dense(46,activation='softmax'))
#kompilacja
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#trenowanie                 
history=model.fit(partial_x_train,
                partial_y_train,
                epochs=20,
                batch_size=512,
                validation_data=(x_val,y_val),
                verbose=0)
trash,results=model.evaluate(partial_x_train,partial_y_train)
print("acc:",results) #0.65 

