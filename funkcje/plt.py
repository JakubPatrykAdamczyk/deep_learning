def plt_make(history):
    import matplotlib.pyplot as plt
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs=range(len(acc))

    plt.plot(epochs,loss,'bo',label='Strata trenowania')
    plt.plot(epochs,val_loss,'b',label='Strata walidacji')
    plt.title('Strata trenowania i walidacji')
    plt.xlabel('epoki')
    plt.ylabel('strata')
    plt.figure()

    #wykres dokładności
    # plt.clf()#clear
    
    plt.plot(epochs,acc,'bo',label='Dokladnosc trenowania')
    plt.plot(epochs,val_acc,'b',label='Dokladnosc walidacji')
    plt.legend( )
    plt.show