import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train: vetores de dados com a base de treino sem as classes
#y_train: vetor com as labels dos vetores presentes no X_train


print(X_train.shape)
#(60000, 28, 28) 60 mil imagens, 28x28 pixels

#Pre-processa os dados, as imagens estao em RGB (3 dimensoes)
#O reshape altera o formato para apenas 1 dimensao

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

#Aqui os valores de 0 a 255 sao normalizados dentro do intervalo [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train[:10])

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print("Y_train.shape = ", Y_train.shape)

## Definindo a arquitetura do modelo

model = Sequential()

## Camada de entrada
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28), 
                        dim_ordering='th'))

## ultimos tres parametros -> (1, 28, 28) (depth, width, height)
## primeiro tres parametros -> (32, 3, 3) 
# (number of convolution filters to use, number of rows in each convolution kernel, 
# number of columns in each convolution kernel)
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Again, we won't go into the theory too much, but it's important to highlight the 
# Dropout layer we just added. This is a method for regularizing our model in order 
# to prevent overfitting. You can read more about it here.
# MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling 
# filter across the previous layer and taking the max of the 4 values in the 2x2 filter.
# So far, for model parameters, we've added two Convolution layers. To complete our model 
# architecture, let's add a fully connected layer and then the output layer:

# For Dense layers, the first parameter is the output size of the layer. Keras automatically 
# handles the connections between layers.
# Note that the final layer has an output size of 10, corresponding to the 10 classes of digits.
# Also note that the weights from the Convolution layers must be flattened (made 1-dimensional) 
# before passing them to the fully connected Dense layer.

## COMPILAR O MODELO

# Para compilar o modelo, deve-se escolher o loss function e o optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Fit model on training data.

# To fit the model, all we have to do is declare the batch size and number 
# of epochs to train for, then pass in our training data.
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=1, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)


outputs = [layer.output for layer in model.layers]
# from matplotlib import pyplot as plt

# fig, axes = plt.subplots(ncols = len(outputs))
# ax = axes.ravel()
# i=0

# for out in outputs:
#     ax[i].imshow(out)
#     ax[i].set_title('Camada: %d' + i)
#     i+=1

# for a in ax:
#     a.axis('off')

# plt.show()

print(outputs[0])

### ARRUMAR AQUI PRA 4 DIMENSIONS
# predict = model.predict(X_test[0])
# print(predict)