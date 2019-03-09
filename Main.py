

# #Train a simple deep CNN on the CIFAR10 small images dataset.
# It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
# (it's still underfitting at that point, though).


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import time
import os

# What GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

dense_layers = [0, 1, 2]
layer_sizes = [16, 32, 64]
conv_layers = [1, 2, 3]

conv_layer_multiplier = 2
dense_layer_multiplier = 16

batch_size = 32
num_classes = 10
epochs = 100
steps_per_epoch = 1563
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# iterate over each combination of number of convolution and dense layers
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}_conv_{}_nodes_{}_dense_{}".format(conv_layer,
                                                         layer_size,
                                                         dense_layer,
                                                         int(time.time()))

            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), padding='same',
                             input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(Conv2D(layer_size, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size * conv_layer_multiplier, (3, 3), padding='same'))
                model.add(Activation('relu'))
                model.add(Conv2D(layer_size * conv_layer_multiplier, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size * dense_layer_multiplier))
                model.add(Activation('relu'))
                model.add(Dropout(0.5))

            model.add(Dense(num_classes))
            model.add(Activation('softmax'))

            # initiate RMSprop optimizer
            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

            # Let's train the model using RMSprop
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

            if not data_augmentation:
                print('Not using data augmentation.')
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          shuffle=True,
                          callbacks=[tensorboard])
            else:
                print('Using real-time data augmentation.')
                # This will do preprocessing and realtime data augmentation:
                datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    zca_epsilon=1e-06,  # epsilon for ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    # randomly shift images horizontally (fraction of total width)
                    width_shift_range=0.1,
                    # randomly shift images vertically (fraction of total height)
                    height_shift_range=0.1,
                    shear_range=0.,  # set range for random shear
                    zoom_range=0.,  # set range for random zoom
                    channel_shift_range=0.,  # set range for random channel shifts
                    # set mode for filling points outside the input boundaries
                    fill_mode='nearest',
                    cval=0.,  # value used for fill_mode = "constant"
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False,  # randomly flip images
                    # set rescaling factor (applied before any other transformation)
                    rescale=None,
                    # set function that will be applied on each input
                    preprocessing_function=None,
                    # image data format, either "channels_first" or "channels_last"
                    data_format=None,
                    # fraction of images reserved for validation (strictly between 0 and 1)
                    validation_split=0.0)

                # Compute quantities required for feature-wise normalization
                # (std, mean, and principal components if ZCA whitening is applied).
                datagen.fit(x_train)

                # Fit the model on the batches generated by datagen.flow().
                model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4,
                                    callbacks=[tensorboard])

            # Save model and weights
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_name = NAME + "_trained_model"
            model_path = os.path.join(save_dir, model_name)
            model.save(model_path)
            print('Saved trained model at %s ' % model_path)

            # Score trained model.
            scores = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
