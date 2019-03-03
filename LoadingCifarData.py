import numpy as np

# This method was given on the CIFAR-10 website, it unpacks the data and puts it into a dict
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

# We want to unpack the first file
batch = unpickle("data_batch_1")
# Let's print out all of the keys so we can see what we're working with (also listed on website)
print(batch.keys())

# Let's save this as a numpy array so we can properly utilize it
img_array = np.array(batch.get(b'data'))

# We can confirm that the shape is the same as on the website (10000, 3072)
print(img_array.shape)