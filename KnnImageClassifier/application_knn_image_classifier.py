# imports
import numpy as np
import struct
from PIL import Image
from flask import Flask, render_template, request
from KNearestNeighbors import KnnClassifier
import json

# initialize application
app = Flask('knn_image_classifier')


# decoder for training labels
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
	return labels


# decoder for training images
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


# home page
@app.route('/')
def upload_files():
	return render_template('upload.html')


# result page after submit	
@app.route('/process', methods=['POST'])
def process_files():
	if request.method == 'POST':

		# read labels and images from training set (IDX file format)
		trainlabels=deconde_idx1_ubyte("mnist_training_data/train-labels.idx1-ubyte")
		trainimages=decode_idx3_ubyte("mnist_training_data/train-images.idx3-ubyte")


		# get uploaded test images (PNG file format)
		uploaded_files = request.files.getlist('files')

		# output dictionary for uploaded images
		dict_test_image_predictions = {}

		for file in uploaded_files:
			itest = np.array(Image.open(file))

			# test image under process
			print(file.filename)
			print('Dimensions of test image -> ')
			print('Number of Rows : ', np.size(itest, 0))
			print('Number of Columns : ', np.size(itest, 1))

			# k = 3 
			classfier = KnnClassifier(1, trainimages, trainlabels, itest, 3)
			predict = classfier.test(1)
			dict_test_image_predictions[file.filename] = str(predict[0])
			print('----------------------------------------------------------------------')


	return str(len(uploaded_files)) + ' images classified successfully.\n' + 'Images vs Digits -> \n' + \
		json.dumps(dict_test_image_predictions)

		
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080)