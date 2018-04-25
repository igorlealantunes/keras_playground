# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.datasets import mnist
#import matplotlib.pyplot as plt
from keras.utils import np_utils
from random import randint

# transforma a saida (ex. 3) em um vetor de 0 ou 1 (ex. [0,0,0,1,0,0,0,0,0,0])
def one_hot_encode(Y):
    T = numpy.zeros((Y.size, 10), dtype=int)
    for idx, row in enumerate(T):
        row[Y[idx]] = 1

    return T

def __main__():

	### leitura do dataset### leit 
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	### Visualizar instâncias
	#plt.imshow(x_train[5], cmap=plt.get_cmap('gray'))
	#plt.title(y_train[5])

	#plt.show()
	

	# 5. Preprocess input data
	x_train = x_train.reshape(x_train.shape[0], 28 * 28)
	x_test = x_test.reshape(x_test.shape[0], 28 * 28)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	 
	# 6. Preprocess class labels
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	print("shape>>>>", x_train.shape)

	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=28*28, activation='relu', kernel_initializer="uniform"))
	model.add(Dense(15, activation='relu', kernel_initializer="uniform"))
	model.add(Dense(10, activation='sigmoid', kernel_initializer="uniform"))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(x_train, y_train, epochs=50, batch_size=500,  verbose=2)
	 
	# test the model
	print("\n\nx_test shape", x_test.shape)
	print("y_test shape", y_test.shape)

	predictions = model.predict(x_test)

	numRes = 35
	randnum = randint(numRes, 10000 - numRes)

	print("\n\n\npredictions", numpy.argmax(predictions[randnum:randnum + numRes], 1))
	print("Ys ------->", y_test[randnum:randnum + numRes].argmax(axis=1))

__main__()

