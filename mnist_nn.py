# Create first network with Keras
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from random import randint

def __main__():

	### leitura do dataset### leit 
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	### Visualizar instâncias
	#plt.imshow(x_train[5], cmap=plt.get_cmap('gray'))
	#plt.title(y_train[5])

	#plt.show()
	
	# Transforma as imagens( arrays 2d) em arrays normais (1 linha - 768 )
	x_train = x_train.reshape(x_train.shape[0], 28 * 28)
	x_test = x_test.reshape(x_test.shape[0], 28 * 28)

	# transforma para float (para depois normalizar)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	# normaliza os dados 
	x_train /= 255
	x_test /= 255
	
	# transforma a saida em inteiros
	y_train = y_train.astype('int8')
	y_test = y_test.astype('int8')


	# muda saida para one hot enconding
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	print("shape>>>>", x_train.shape)

	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=28*28, activation='sigmoid', kernel_initializer="uniform"))
	model.add(Dense(15, activation='tanh', kernel_initializer="uniform"))
	model.add(Dense(20, activation='relu', kernel_initializer="uniform"))
	model.add(Dense(10, activation='softmax', kernel_initializer="uniform"))
	# Compile model

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	history = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.10, verbose=1)
	 
	# test the model
	print("\n\nx_test shape", x_test.shape)
	print("y_test shape", y_test.shape)

	predictions = model.predict(x_test)

	numRes = 35
	randnum = randint(numRes, 10000 - numRes)

	print("\n\n\npredictions", numpy.argmax(predictions[randnum:randnum + numRes], 1))
	print("Ys ------->", y_test[randnum:randnum + numRes].argmax(axis=1))

	# history

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


__main__()

