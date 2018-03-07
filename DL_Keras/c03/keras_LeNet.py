# import the necessary packages
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import platform

np.random.seed(1671)  # for reproducibility

# define the convnet. Yann LeCunn
class LeNet:
	@staticmethod
	def build(input_shape, classes):
		model = Sequential()

		'''
		# Conv2D( filter 갯수, kernel_size : 합성곱 크기, padding="same" )
		# padding = "same" 	-- 입력과 출력의 크기가 동일. 가장자리에 0을 채운다.
		# padding = "valid"	-- 출력이 입력보다 작아짐.
		# deep layer의 filter 수 증가시킴 : 20 --> 50
		'''

		# kernel_size=5 는 (5, 5) 와 동일
		model.add(Conv2D(20, kernel_size=5, padding="same", activation='relu', input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(50, kernel_size=5, padding="same", activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# flatten
		model.add(Flatten())
		model.add(Dense(500, activation='relu'))

		# softmax classifier
		model.add(Dense(classes, activation='softmax'))

		return model


# network and training
NB_EPOCH = 20
BATCH_SIZE = 1000  # 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.1

IMG_ROWS, IMG_COLS = 28, 28 			# input image dimensions
NB_CLASSES = 10  						# number of outputs = number of digits
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)

# tf : ( rows, cols, dim )
# th : ( dim, rows, cols )
K.set_image_dim_ordering("th") # theano


# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# consider them as float and normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# reshape : 60,000 x [1 x 28 x 28]
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)


# initialize the optimizer and model
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

history = model.fit(X_train, y_train,
					batch_size=BATCH_SIZE, epochs=NB_EPOCH,
					verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, y_test)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
print('')

if platform.system() == 'Windows':
	import matplotlib.pyplot as plt

	# list all data in history
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
else:
	print('acc', history.history['acc'])
	print('val_acc',history.history['val_acc'])
