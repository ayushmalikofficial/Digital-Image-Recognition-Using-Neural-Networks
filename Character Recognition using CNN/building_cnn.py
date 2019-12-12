#Dataset Description
#Modified NIST (MNIST) has 3D arrays
#Each image is a 28 by 28 pixel square (784 pixels total)
#Training Set has 60,000 images has dimensions (60,000,28,28)
#Testing Set has 10,000 images has dimensions (10,000,28,28)
#Keras API is used for CNN

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#"th" format means that the convolutional kernels will have the shape (depth, input_depth, rows, cols)
#"tf" format means that the convolutional kernels will have the shape (rows, cols, input_depth, depth)

K.set_image_dim_ordering('th')


#Initialize Random No generator
#So that We can reproduce identical results
seed = 7
numpy.random.seed(seed)


#Loading dataset automatically at backend with the help of KERAS
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('MNIST data set loaded ')
#CNN needs a 4D array as input [batch][pixels][width][height]
#Reshaping to [batch][pixels][width][height]
# 1 denotes grayscale other wise we would have 3 for colored images
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')



#Normalization
x_train = x_train / 255
x_test = x_test / 255

#Note we will convert our prediction range 0-9 into a binary array
#For eg 8 means [0,0,0,0,0,0,1,0]
#This is also called 'one-hot encoding'
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]





#*******************************CREATING THE MODEL***********************************

#Function to create a CNN
def build_cnn():
	#Creating a sequential model in KERAS
        #Sequential model is a liner stack of layers
        #We can create the model by passing a list of layer instances to the constructor Sequential or use the method which we have used.

        #Calling Constructor
	model = Sequential()
        #model is a object of Sequential
        #model.add() adds a layer to our Keras Model object i.e model
	
        
	#First hidden layer Convolution Layer. It has 32 feature maps of size 5 x 5 and activation function relu
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        #Max Pooling Layer.Pool Size = 2 x 2 
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Randomly excludes 20% neurons to reduce overfitting
	model.add(Dropout(0.2))
	#Flattens the 2D matrix to a vector
	model.add(Flatten())
	#Fully Connected Layer (No. of Neurons : 128 )
	model.add(Dense(128, activation='relu'))
	#10 Neuron O/P
	#Softmax is an activation Function. Analogous to sigmoid in binary classification
        #Each neuron will give the probability of that class
	model.add(Dense(num_classes, activation='softmax'))
	#Creating a model
	#Logarithmic Loss (categorical_crossentropy) : -[y.log(a)+(1-y).log(1-a)]
	#Gradient Descent : ADAM ( a method of stochastic optimization )
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#Function Call
model = build_cnn()
print('Model Built')





#*********************************TRAINING THE MODEL***************************************


print('*************************TRAINING STARTED******************************')
print('\nEpochs : 15')
print('Batch Size : 200')
print()
#.fit function is used to train
#fit(x, y, batch_size, epochs, verbose)
#x: Numpy array of training data 
#y: Numpy array of target (label) data
#batch_size: Number of samples per gradient update.
#epochs:Number of epochs to train the model. An epoch is an iteration over the entire  x and y data provided.
#verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=200, verbose=2)


#Saving the model
model.save('models/CNN.h5')
print('\nModel Saved')





#****************************Final evaluation of the model******************************

print('*************************EVALUATING******************************')

#.evaluate returns the loss value & metrics values
#batch_size: Number of samples per evaluation step. If unspecified, batch_size will default to 32.
#verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
scores = model.evaluate(x_test, y_test, verbose=0)

#Calculating Efficiency
print("\nConvolutional Neural Network Error: %.2f%%" % (100-scores[1]*100))
