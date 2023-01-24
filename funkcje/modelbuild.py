def build_model(train_data,loss,metrics,optimizer='rmsprop'):
    from keras import models,layers
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer,
                loss=loss,#średnia błędu kwadratowego
                metrics=[metrics])#średni błąd bezwzględny
    return model