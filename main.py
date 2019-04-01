import tensorflow as tf
# import numpy as np
# from PIL import Image
# import skimage
# import h5py

# from keras.preprocessing import image
# from matplotlib import pyplot as plt
from keras.layers import Input, Conv2D
from keras.models import Model, Sequential

tf.io.decode_image(
    'image.jpg',
    channels=None,
    dtype=tf.dtypes.uint8,
    name=None
)

def simple_model(input_shape):

    model = Sequential()

    model.add(Conv2D(32, (7, 7), input_shape=input_shape))

    # x_input = Input(input_shape)
    # x = Conv2D(1, (7, 7), padding='valid', strides=(2, 2), name='conv0')(x_input)
    #
    # model = Model(inputs=x_input, outputs=x, name='simple_model')

    return model


# generator = datagen.flow_from_directory(
#         'data/train',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode=None,  # this means our generator will only yield batches of data, no labels
#         shuffle=False)
#
# img = image.load_img('image.jpg', target_size=(64, 64, 3))
#
# m = simple_model((64, ))
#
# input = Input((64, 64, 3))
# input = Conv2D(1, (7, 7), padding='valid', strides=(2, 2), name='conv0')(input)
#
# print(input)

# plt.imshow(img)
# plt.show()


# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./image.jpg"))
# image_reader = tf.WholeFileReader()
#
# _, image_file = image_reader.read(filename_queue)
# image = tf.image.decode_jpeg(image_file)
#
# with tf.Session() as sess:
#     # Required to get the filename matching to run.
#     tf.global_variables_initializer().run()
#
#     # Coordinate the loading of image files.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     # Get an image tensor and print its value.
#     image_tensor = sess.run([image])
#     print(image_tensor)
#
#     # Finish off the filename queue coordinator.
#     coord.request_stop()
#     coord.join(threads)


# image2 = tf.image.resize_images(
#     image,
#     (64, 64)
# )
#
# image2.show()