from keras import layers,models,optimizers

# input_tensor=layers.Dense(32,activation='relu',input_shape=(784,))
# target_tensor=layers.Dense(10,activation='softmax')
# model=models.Sequential()
# model.add(input_tensor)
# model.add(target_tensor)
# to samo tylko api
input_tensor=layers.Input(shape=(784,))
x=layers.Dense(32,activation='relu')(input_tensor)
target_tensor=layers.Dense(10,activation='softmax')(x)
model=models.Model(inputs=input_tensor,outputs=target_tensor)

model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
    loss='mse',
    metrics=['accuracy'])



# model.fit(input_tensor,target_tensor,batch_size=128,epochs=10)

