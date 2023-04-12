from PIL import Image
import tensorflow as tf
import numpy
from resize import data, testdata
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

#labels - screenshot
#features - player 0-1

#x_train = [] #screen shots - Features
#y_train = [] #palyer 1 = yes, 0 = no - Labels

(x_train,y_train) = data()
x_train = numpy.array(x_train)/255
y_train = numpy.array(y_train)/255
print(x_train.shape,y_train.shape)


(x_test,y_test) = testdata()
x_test = numpy.array(x_test)/255
y_test = numpy.array(y_test)/255

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(360,640,3)),
  tf.keras.layers.Conv2D(64,(3,3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Conv2D(64,(3,3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Conv2D(64,(3,3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Conv2D(64,(3,3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Conv2D(64,(3,3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(5184, activation='relu'),
  tf.keras.layers.Dense(2592, activation='relu'),
  tf.keras.layers.Dense(1296, activation='relu'),
  tf.keras.layers.Dense(648, activation='relu'),
  tf.keras.layers.Dense(648, activation='relu'),
  tf.keras.layers.Dense(324, activation='relu'),
  tf.keras.layers.Dense(162, activation='relu'),
  tf.keras.layers.Dense(81, activation='relu'),
  tf.keras.layers.Dense(27, activation='relu'),
  tf.keras.layers.Dense(9, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer='SGD',
              metrics=['Accuracy'])

model.fit(x_train, 
          y_train,
          batch_size = 1,
          shuffle = True,
          verbose = 1,
          epochs=50)

results = model.evaluate(x_test, y_test, batch_size=32)
print("EVAL: ", results)
print("0 - no enemy,     1 - enemy")


model.save("model")