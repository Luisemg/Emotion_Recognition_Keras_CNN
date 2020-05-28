import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

num_classes = 3
img_rows, img_cols = 48,48
batch_size = 128

train_data_dir = 'NHS_DataSet/train'
validation_data_dir = 'NHS_DataSet/validation'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    color_mode='grayscale',
                    target_size=(img_rows,img_cols),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
                        validation_data_dir,
                        color_mode='grayscale',
                        target_size=(img_rows,img_cols),
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=True)

model = Sequential()

model.add(Conv2D(32,(3,3),padding='same', kernel_initializer='he_normal', activation='elu', input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same', kernel_initializer='he_normal', activation='elu', input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(64,(3,3),padding='same', activation='elu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same', activation='elu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128,(3,3),padding='same', activation='elu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same', activation='elu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256,(3,3),padding='same', activation='elu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same', activation='elu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(64,activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64,activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(num_classes, kernel_initializer='he_normal', activation='softmax'))

print(model.summary())

checkpoint = ModelCheckpoint(
    'NHS_vgg.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_delta=0.0001)

callbacks = [checkpoint, reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.003),
              metrics=['accuracy'])

nb_train_samples = 15421
nb_validation_samples = 1921
epochs = 10

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples//batch_size)

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
