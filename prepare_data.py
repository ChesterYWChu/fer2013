import numpy as np
import pandas as pd
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

if __name__ == '__main__':
	X = []
	df = pd.read_csv(sys.argv[1])
	for index, row in df.iterrows():
		if row['emotion'] == 1 and row['Usage'] == 'Training':
			X.append([int(p) for p in row['pixels'].split()])

	X = np.array(X)
	X = X.reshape(X.shape[0], 48, 48, 1)
	print(X.shape)

	datagen = ImageDataGenerator(
	        rotation_range=30,
	        width_shift_range=0.1,
	        height_shift_range=0.1,
	        shear_range=0.2,
	        zoom_range=0.2,
	        horizontal_flip=True,
	        fill_mode='nearest')

	aug_X = None
	i = 0
	# for batch in datagen.flow(X, batch_size=1, save_to_dir='tmp', save_prefix='emotion_1_', save_format='jpeg'):
	for batch in datagen.flow(X, batch_size=1):
		if aug_X is None:
			aug_X = batch.reshape(1, 48*48)
		else:
			aug_X = np.concatenate((aug_X, batch.reshape(1, 48*48)), axis=0)

		i += 1
		if i >= (X.shape[0] * 9):
			break

	aug_X = aug_X.astype(int)

	for x in aug_X:
		if x.shape != (48*48,):
			print('1 Wrong number of pixels!')
			print(x.shape)

	# aug_X = np.apply_along_axis(lambda d: [' '.join(str(x) for x in d)], 1, aug_X)
	# aug_X = np.apply_along_axis(lambda d: [x.strip() for x in d], 1, aug_X)

	tmp = []
	for x in aug_X:
		tmp.append([' '.join(str(p) for p in x)])

	aug_X = np.array(tmp)
	print(aug_X.shape)

	# np.savetxt('outout', aug_X, delimiter=',', fmt='%s')
	for x in aug_X:
		if len(x[0].split(' ')) != 2304:
			print('2 Wrong number of pixels!')
			print(len(x[0].split(' ')))

	emotion_col = np.full((aug_X.shape[0], 1), 1)
	Usage_col = np.full((aug_X.shape[0], 1), 'Training')
	data = np.append(emotion_col, aug_X, axis=1)
	data = np.append(data, Usage_col, axis=1)
	print(data.shape)

	np.savetxt('augmented_data.csv', data, delimiter=',', fmt='%s')




