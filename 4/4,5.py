from keras import regularizers,models,layers
number_tent=10000

#technika porzucenia imbd
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(number_tent,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
#porzucenie pozwala nam zmniejszyć ilość strat w walidacji 