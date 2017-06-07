from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import numpy as np
import sys
import PIL

# Usage: python predict.py classifiers/classifier_batch200_augmented_val_acc_0.5305.h5 data/fer2013_private_test.csv tmp

batch_size = 1
img_width = 48
img_height = 48

if __name__ == "__main__":
	print('Load model:' + sys.argv[1])
	model = load_model(sys.argv[1])
	model.summary()  #Verifying model structure

	df = pd.read_csv(sys.argv[2])
	X = df['pixels'].as_matrix()
	Y = df['emotion'].as_matrix()

	tmp = []
	for x in X:
		tmp.append([int(p) for p in x.split(' ')])
	X = np.array(tmp)
	X = X.reshape(X.shape[0], img_width, img_height, 1)
	print(X.shape)

	i = 0
	datagen = ImageDataGenerator()
	for batch in datagen.flow(X, batch_size=1, save_to_dir=sys.argv[3]+'/tmp', save_prefix='img', save_format='jpeg'):
		i += 1
		if i >= (X.shape[0]):
			break




	test_datagen = ImageDataGenerator(rescale=1. / 255)
	test_generator = test_datagen.flow_from_directory(
	    sys.argv[3],
	    target_size=(img_width, img_height),
	    batch_size=batch_size,
	    color_mode='grayscale',
	    class_mode=None)



	predictions = model.predict_generator(test_generator, X.shape[0])
	print(predictions)
	print(Y)

