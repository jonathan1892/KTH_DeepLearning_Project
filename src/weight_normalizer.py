# Normalizes the weights in the VGG19 network by running through the images in the
# ILSVRC2012 validation set in batches, capturing the activation, and normalizing the output
from PIL import Image
import math
import numpy as np
import time
import glob
from keras import backend as K
from keras.models import Model
from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import gc
from ScipyOptimizer import ScipyOptimizer
from datetime import datetime

# Adjust path to download location for the image set
content_paths = [filename.replace('\\', '/') for filename in glob.glob('d:/DeepLearning/Data/ILSVRC2012_img_val/*.JPEG')]

# Network related
meanRGB = [123.68, 116.779, 103.939]
width = 512
height = 512

# Transforms an image object into an array ready to be fed to VGG
def preprocess_image(image):
    image = image.resize((height, width))
    array = np.asarray(image, dtype="float32")
    if len(array.shape) != 3:
        new_array = np.expand_dims(array, axis=3)
        # Adjust for greyscale images
        array = np.tile(new_array, 3)
    array = np.expand_dims(array, axis=0) # Expanding dimensions in order to concatenate the images together

    array[:, :, :, 0] -= meanRGB[0] # Subtracting the mean values
    array[:, :, :, 1] -= meanRGB[1]
    array[:, :, :, 2] -= meanRGB[2]
    array = array[:, :, :, ::-1] # Reordering from RGB to BGR to fit VGG19
    return array


# Transforms an array representing an image into a scipy image object
def deprocess_array(array):
    deprocessed_array = np.copy(array)
    deprocessed_array = deprocessed_array.reshape((height, width, 3))
    deprocessed_array = deprocessed_array[:, :, ::-1] # BGR to RGB
    deprocessed_array[:, :, 0] += meanRGB[0]
    deprocessed_array[:, :, 1] += meanRGB[1]
    deprocessed_array[:, :, 2] += meanRGB[2]
    deprocessed_array = np.clip(deprocessed_array, 0, 255).astype("uint8")
    image = Image.fromarray(deprocessed_array)
    return image

def load_content_array(content_path):
    #Print progress every 1000 images
    if "000.JPEG" in content_path:
        print(content_path + ', time ' + str(datetime.now()))

    content_image = Image.open(content_path)
    if content_image.mode == 'CMYK':
        content_image = content_image.convert('RGB')

    content_array = preprocess_image(content_image)
    return content_array

###### Model Loading
model = VGG19(input_tensor=None, weights="imagenet", include_top=False, pooling="avg")

model_layers = dict([(layer.name, layer.output) for layer in model.layers])
conv_layer_names = [layer for layer in sorted(model_layers.keys()) if 'conv' in layer]
content_count = len(content_paths)

batch_size = 10000
batches = (len(content_paths) + batch_size - 1) // batch_size
first_initialized = False
batch_direction = -1
for layer_name in conv_layer_names:
    output_layer = model.get_layer(layer_name)
    get_layer_output = K.function([model.layers[0].input],
                                   [output_layer.output])
    mean_total = 0
    item_counter = 0
    if first_initialized == False:
        content_arrays = []
#    content_arrays2 = []
    batch_direction = -1 * batch_direction
    first_batch = True
    for batch in range(1, batches + 1):
        if batch > 1:
            first_batch = False
        if batch_direction == -1:
            batch = batches - batch + 1
        print('layer '+ layer_name + ', batch ' + str(batch) + ', time ' +  str(datetime.now()))
        if first_batch and not first_initialized:
            del content_arrays
            gc.collect()
            content_arrays = [load_content_array(content_path) for content_path in content_paths[(batch - 1)*batch_size: batch * batch_size]]
        elif not first_batch:
            del content_arrays
            gc.collect()
            content_arrays = [load_content_array(content_path) for content_path in content_paths[(batch - 1)*batch_size: batch * batch_size]]

        first_initialized = True
#        content_arrays2 = content_arrays
#        for content_array in content_arrays2:
        for content_array in content_arrays:
            item_counter = item_counter + 1
            layer_output = get_layer_output([content_array])[0]
            activation_count_per_filter = layer_output.shape[1]*layer_output.shape[2]
            #layer_output in conv1_1 is 1, 512, 512, 64
            activation_per_filter = np.sum(layer_output, axis=1)
            activation_per_filter = np.sum(activation_per_filter, axis=1)
            mean = activation_per_filter / activation_count_per_filter
            mean_total = mean_total + (mean - mean_total) / item_counter
            #each filter outputs 512x512 in conv_1_1, same as input size.
    filter_scale_factor = 1 / mean_total
    filter_scale_factor = filter_scale_factor[0]
    layer_weights = output_layer.get_weights()
    #3x3x64 in conv1_1
    filter_weights = layer_weights[0]
    filter_bias = layer_weights[1]

    filter_weights_rescaled = filter_scale_factor * filter_weights
    bias_weights_rescaled = filter_scale_factor * filter_bias

    layer_weights[0] = filter_weights_rescaled
    layer_weights[1] = bias_weights_rescaled
    output_layer.set_weights(layer_weights)
    print("Layer completed: " + layer_name)

#save
model_json = model.to_json()
with open("../models/normalized.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../models/normalized.h5")
print("Saved model to disk")
