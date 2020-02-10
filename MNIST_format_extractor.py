# Reads a File in the MNIST Format (.gz) and extracts its contents to be processed

import gzip
import numpy as np

# Helper function to read each file
def read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder(">")
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

# file parameter to be in the format of TensorFlow's GFile
def extract_images(file):
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic = read32(bytestream)
        if magic != 2051:
            raise ValueError("Invalid magic number " + str(magic) + " in MNIST-Format image file: " + str(file.name))
        numImages = read32(bytestream)
        rows = read32(bytestream)
        columns = read32(bytestream)
        buffer = bytestream.read(rows * columns * numImages)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(numImages, rows, columns, 1)
        assert data.shape[3] == 1
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2])    # (numImages, rows, columns)
        data = data.astype(np.float32)
        data = np.multiply(data, 1.0 / 255.0)
        data = np.swapaxes(data,1,2)    # Swaps pixel axes to make the image right-up
        return data

# file parameter to be in the format of TensorFlow's GFile
def extract_labels(file):
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic  = read32(bytestream)
        if magic != 2049:
            raise ValueError("Invalid magic number " + str(magic) + " in MNIST-Format image file: " + str(file.name))
        numItems = read32(bytestream)
        buffer = bytestream.read(numItems)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

