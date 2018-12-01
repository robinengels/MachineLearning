#!/usr/bin/env python

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(20, (5, 5), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))

classifier.add(Conv2D(50, (5, 5), padding="same",activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 500, activation = 'relu'))

classifier.add(Dense(units = 6, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('test',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

classifier.fit_generator(training_set,
steps_per_epoch = 800,
epochs = 6,
validation_data = test_set,
validation_steps = 2000)

classifier.save("saved_model.h5")
