import numpy as np
import cv2
import os
import struct
import Knndraft

def deconde_idx1_ubyte(filename):
    bin_data = open(filename, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    print ("label", labels)
    return labels

def decode_idx3_ubyte(filename):
    bin_data = open(filename, 'rb').read()

    offset = 0
    fmt_header = '>iiii' 
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  
    fmt_image = '>' + str(image_size) + 'B' 
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #images[i]=np.where(images[i]>175,1,0)       
        offset += struct.calcsize(fmt_image)
    images = np.reshape(images, (num_images*28, 28))
    return images


trainlabels=deconde_idx1_ubyte("./train-labels.idx1-ubyte")
trainimages=decode_idx3_ubyte("./train-images.idx3-ubyte")
testlabels=deconde_idx1_ubyte("./t10k-labels.idx1-ubyte")
testimages=decode_idx3_ubyte("./t10k-images.idx3-ubyte")
classfier=Knndraft.KnnClassifier(1, trainimages, trainlabels, testimages, 3)
predict=classfier.test(1)
print("lenth of predict:",len(predict))
correct=0
for i in range(len(predict)):
    if predict[i]==testlabels[i]:
    	correct+=1
print("correctness:", correct/len(predict))
