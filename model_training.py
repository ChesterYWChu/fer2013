import numpy as np
import pandas as pd
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.utils import to_categorical


# Usage: python model_training.py data/fer2013_training_aug.csv data/fer2013_public_test.csv

img_width, img_height = 48, 48
nb_train_samples = 32633
nb_validation_samples = 3589
epochs = 50
batch_size = 200

def load_data(file):
	X = []
	Y = []
	head = True
	for line in open(file):
		if head:
			head = False
		else:
			col = line.split(',')
			Y.append(int(col[0]))
			X.append([int(p) for p in col[1].split()])

	X, Y = np.array(X), np.array(Y)

	return X, Y

def to_image(X):
	n, d = X.shape
	d = int(np.sqrt(d))
	# X = X.reshape(n, 1, d, d)
	X = X.reshape(n, d, d, 1)
	return X

def show_image(row):
	print(row)
	print(row.shape)
	plt.imshow(row[0], cmap='gray')
	plt.show()

def get_train_generator(X, Y):
	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

	train_generator = train_datagen.flow(
	        X,
	        Y,
	        batch_size=batch_size) 
	return train_generator

def get_test_generator(X_test, Y_test):
	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow(
	        X_test,
	        Y_test,
	        batch_size=batch_size) 
	return test_generator

if __name__ == '__main__':
	X, Y = load_data(sys.argv[1])
	X = to_image(X)
	Y = to_categorical(Y, num_classes=7)
	train_generator = get_train_generator(X, Y)

	X_test, Y_test= load_data(sys.argv[2])
	X_test = to_image(X_test)
	Y_test = to_categorical(Y_test, num_classes=7)
	test_generator = get_test_generator(X_test, Y_test)

	if K.image_data_format() == 'channels_first':
	    input_shape = (1, img_width, img_height)
	else:
	    input_shape = (img_width, img_height, 1)
	print(input_shape)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(7))
	model.add(Activation('softmax'))



	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)

	# model.fit(
	# 	X,
	# 	Y,
	# 	epochs=epochs,
	# 	batch_size = batch_size,
	# 	validation_split=0.2,
	# 	callbacks=[early_stopping_monitor]
	# 	)

	model.fit_generator(
		train_generator,
		epochs=epochs,
		steps_per_epoch=nb_train_samples // batch_size,
		validation_data=test_generator,
		validation_steps=nb_validation_samples // batch_size,
		callbacks=[early_stopping_monitor]
		)

	model.save('classifier.h5')







