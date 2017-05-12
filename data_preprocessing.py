import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import os
from scipy import misc # run "pip install pillow"

#http://www.scipy-lectures.org/advanced/image_processing/

people_small = ['F01','F02']
people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
data_types = ['phrases', 'words']
folder_enum = ['01','02','03','04','05','06','07','08','09','10',]

VALIDATION_SPLIT = ['F07']
TEST_SPLIT = ['F11']

X_train = None
y_train = None

X_val = None
y_val = None

X_test = None
y_test = None  

for person_ID in people_small:
	for phrase_ID in folder_enum:
		for instance_ID in folder_enum:
			for data_type in data_types:
				directory = person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'
				filelist = os.listdir(directory)

				for img_name in filelist:
					if img_name.startswith('color'):
						image = misc.imread(directory + '' + img_name)

						print image.shape

				
				# # Validation data
				# if person_ID in VALIDATION_SPLIT:
				# 	X_val 
				# 	y_val

				# # Test data
				# if person_ID in TEST_SPLIT:
				# 	X_test
				# 	y_test

				# # Train data 
				# else: 
				# 	X_train
				# 	y_train



					











