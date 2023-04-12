import tensorflow as tf
import numpy
from resize import testdata

(test_images, test_labels) = testdata()
test_images = numpy.array(test_images)
test_labels = numpy.array(test_labels)
print(test_images.shape,test_labels.shape)

model = tf.keras.models.load_model('model')
model.summary()

model.evaluate(test_images, test_labels, verbose=2)

