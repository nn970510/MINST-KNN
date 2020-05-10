import numpy as np
import cv2
import os
import struct
import Knndraft
import sys

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser  
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-t', '--training', help=default('The percent of the training set'), default=100, type="int")
    parser.add_option('-d', '--subsample', help=default('Whether to subsample pic'), default=0, type="int")
    parser.add_option('-k', '--knum', help=default("Knn's K number"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Percent of test data to use"), default=100, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}
    # Set up variables according to the command line input.
    print ("Knn's K is:\t" + str(options.knum))
    print ("training set size:\t" + str(options.training*60000/100))
    print ("test set size:\t" + str(options.test*10000/100))
    print ("Use subsample or not:\t" + str(options.subsample))
    
    if options.training <= 0 or options.training>100:
      print ("Training size percent should between 0-100 (you provided: %d)" % options.training)
      print (USAGE_STRING)
      sys.exit(2)

    if options.test <= 0 or options.test>100:
      print ("Test size percent should between 0-100 (you provided: %d)" % options.test)
      print (USAGE_STRING)
      sys.exit(2)
    
    if options.knum <= 0 or options.knum>=10:
      print ("Please provide a 0-10 numbers for k (you provided: %f)" % options.knum)
      print (USAGE_STRING)
      sys.exit(2)

    return options

USAGE_STRING = """"""

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
options=readCommand(sys.argv[1:])
testimages=testimages[:int(len(testimages)*(options.test/100))]
classfier=Knndraft.KnnClassifier(options.training/100, trainimages, trainlabels, testimages, options.knum)
predict=classfier.test(options.subsample)
print("lenth of predict:",len(predict))
correct=0
for i in range(len(predict)):
    if predict[i]==testlabels[i]:
    	correct+=1
print("correctness:", correct/len(predict))
