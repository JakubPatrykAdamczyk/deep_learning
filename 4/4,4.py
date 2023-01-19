from keras import regularizers,models,layers
number_tent=10000

#regularyzacja

regularizers.l1(0.001)          #l1
regularizers.l2(0.001)          #l2
regularizers.l1_l2(0.001)       #jednocześnie l1 i l2

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(number_tent,),kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1,activation='sigmoid'))


