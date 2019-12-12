#Dataset Description
#Modified NIST (MNIST) has 3D arrays
#Each image is a 28 by 28 pixel square (784 pixels total)
#Training Set has 60,000 images has dimensions (60,000,28,28)
#Testing Set has 10,000 images has dimensions (10,000,28,28)

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#Loading dataset automatically at backend with the help of KERAS
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('MNIST data set loaded ')


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#Note we will convert our prediction range 0-9 into a binary array
#For eg 8 means [0,0,0,0,0,0,1,0]
#This is also called 'one-hot encoding'
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


#*******************************CREATING THE MODEL***********************************

#Function to create a ANN
def build_ann():

        #Creating a sequential model in KERAS
        #Sequential model is a liner stack of layers
        #We can create the model by passing a list of layer instances to the constructor Sequential or use the method which we have used.

        #Calling Constructor
	
	model = Sequential()
	#model is a object of Sequential
        #model.add() adds a layer to our Keras Model object i.e model
	
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))


	#Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
#Function Call
model = build_ann()
print('Model Built')
#*********************************TRAINING THE MODEL***************************************


print('*************************TRAINING STARTED******************************')
print('\nEpochs : 10')
print('Batch Size : 200')
print()
#.fit function is used to train
#fit(x, y, batch_size, epochs, verbose)
#x: Numpy array of training data 
#y: Numpy array of target (label) data
#batch_size: Number of samples per gradient update.
#epochs:Number of epochs to train the model. An epoch is an iteration over the entire  x and y data provided.
#verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

#Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

#Saving the model
model.save('models/ANN.h5')
print('\nModel Saved')



#****************************Final evaluation of the model******************************

print('*************************EVALUATING******************************')

#.evaluate returns the loss value & metrics values
#batch_size: Number of samples per evaluation step. If unspecified, batch_size will default to 32.
#verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
scores = model.evaluate(X_test, y_test, verbose=0)

#Calculating Efficiency
print("\nConvolutional Neural Network Error: %.2f%%" % (100-scores[1]*100))




