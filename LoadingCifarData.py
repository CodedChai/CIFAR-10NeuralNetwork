import numpy as np
from matplotlib import pyplot as plt
import pickle

FILENAMES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

# This method was given on the CIFAR-10 website, it unpacks the data and puts it into a dict
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict


img_array = []
# Label, this is what the classification actually is
y = []
# We want to unpack all of the files
for filename in FILENAMES:
    batch = unpickle(filename)
    # Let's print out all of the keys so we can see what we're working with (also listed on website)
    print(batch.keys())

    # We will flatten the numpy array and store it as a list then reconvert it to a numpy array after we load in all of the data
    img_array.append(np.array(batch.get(b'data')).flatten())
    y.append(batch[b'labels'])

# We want to reshape the flattened data from the numpy array
img_array = np.array(img_array).reshape(-1, 3072)
# We can confirm that the shape is the same as on the website (10000 * 5, 3072)
print(img_array.shape)

# Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
# The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image. 

# This is going to be our features
X = []

# First we will split the numpy array into the separate images
for image in img_array:
    # Now we can split it up into the specific RGB values, I chose to do this so the RGB values are grouped right next to each other, this will make it easier to test against my own images
    for index in range(1024):
        X.append(image[index])
        X.append(image[index * 2])
        X.append(image[index * 3])

# Features, basically just the RGB data
X = np.array(X).reshape(-1, 32, 32, 3)

# Let's reconfirm that the shape looks right (any number of images, 32 pixels by 32 pixels with 3 RGB values)
print(X.shape)

# Finally let's save this data for easy access later

pickle_out = open("CIFAR_IMAGE_FEATURES.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("CIFAR_LABELS.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()