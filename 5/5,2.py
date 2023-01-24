#dataset on /kaggle/input/dogs-vs-cats
#https://www.kaggle.com/code/jakubpatrykadamczyk/cat-vs-dog
import os,shutil



import zipfile
path='/kaggle/input/dogs-vs-cats'
work_dir='/kaggle/working'

z= zipfile.ZipFile(path+'/train.zip')
z.extractall()
train_dir='/kaggle/working/train'


z2= zipfile.ZipFile(path+'/test.zip')
z2.extractall()
test_dir='/kaggle/working/test1'

validation_dir=os.path.join(work_dir,'validation')
os.mkdir(validation_dir)

train_cat_dir=os.path.join(train_dir,'cat')
os.mkdir(train_cat_dir)

train_dog_dir=os.path.join(train_dir,'dog')
os.mkdir(train_dog_dir)

validation_cat_dir=os.path.join(validation_dir,'cat')
os.mkdir(validation_cat_dir)

validation_dog_dir=os.path.join(validation_dir,'dog')
os.mkdir(validation_dog_dir)

test_cat_dir=os.path.join(test_dir,'cat')
os.mkdir(test_cat_dir)

test_dog_dir=os.path.join(test_dir,'dog')
os.mkdir(test_cat_dir)

#kopiujemy zdjęcia treningowe do plików
for file in os.listdir(train_dir):
    if file[-3]=='j':
        if file[0]=='c':
            os.replace(os.path.join(train_dir,file),os.path.join(train_cat_dir,file))
        else:
            os.replace(os.path.join(train_dir,file),os.path.join(train_dog_dir,file))

#kopiujemy z test do validation
fnames=['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(test_dir,fname)
    dst=os.path.join(validation_dog_dir,fname)
    shutil.copyfile(src,dst)
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(test_dir,fname)
    dst=os.path.join(validation_cat_dir,fname)
    shutil.copyfile(src,dst)

#sprawdzamy ilość zdjęć kotów
cat_train_len=len(os.listdir(train_cat_dir))
print(cat_train_len)

dog_train_len=len(os.listdir(train_dog_dir))
print(dog_train_len)


#Tworzymy sieć
#from tensorflow.keras import layers,models,optimizers
from keras import layers,models,optimizers

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#kompilacja
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

#wczytanie obrazów
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)#przeskalowanie 1/255
validation_datagen=ImageDataGenerator(rescale=1./255)#przeskalowanie 1/255

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator=validation_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

#trening
history=model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=0)
model.save('cat_and_dog.h5')

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

