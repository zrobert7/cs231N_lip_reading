import os
from scipy import misc
import numpy as np

people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
data_types = ['words']

folder_enum = ['01','02','03','04','05','06','07','08','09','10']

max_width = 0
max_height = 0

min_width = 200
min_height = 200

directory = '../cropped'
for person_id in people:
	for word_index, word in enumerate(folder_enum):
		for iteration in folder_enum:
			path = os.path.join(directory, person_id, 'words', word, iteration)
			filelist = os.listdir(path + '/')
			sequence = []
			for img_name in filelist:
				if img_name.startswith('color'):
					image = misc.imread(path + '/' + img_name)
					w,h,c = np.shape(image)
					if w > max_width:
						max_width = w
					if w < min_width:
						min_width = w
					if h > max_height:
						max_height = h
					if h < min_height:
						min_height = h	
					#print('path' + '/' + img_name + ': ' + str(np.shape(image)))
	print('finished person ' + person_id)

print('max height: ' + str(max_height))
print('max width: ' + str(max_width))
print('min height: ' + str(min_height))
print('min width: ' + str(min_width))


# Permissions on folders
# Github permissions