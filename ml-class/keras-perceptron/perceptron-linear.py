from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
img_width = X_train.shape[1]
img_height = X_train.shape[2]

print(y_train.shape) # (60000,)
# exit()

# one hot encode outputs
y_train = to_categorical(y_train) # one hot encoding
y_test = to_categorical(y_test)
labels = range(10)

num_classes = y_train.shape[1] # this returns 10, due to reassign

print(num_classes) 
exit()

# create model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(num_classes, activation='sigmoid')) # 785 * 10
"""
sigmoid =  fix input 0 - 1 
softmax = fix input 0 - 1 and all values some to 1
"""

model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

# uses identity function if you don't specify activation function

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])
model.save('model.h5')
