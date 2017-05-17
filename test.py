from scipy import misc
import numpy as np

image = misc.imread('cropped/F02/words/09/10/color_001.jpg')
print(np.shape(image))


people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
data_types = ['words']

folder_enum = ['01','02','03','04','05','06','07','08','09','10']

directory = 'cropped'
for person_id in people:
	for word_index, word in enumerate(folder_enum):
		for iteration in folder_enum:
			path = os.path.join(directory, person_id, 'words', word, iteration)
			filelist = os.listdir(path + '/')
			sequence = []
			for img_name in filelist:
				if img_name.startswith('color'):
					image = misc.imread(path + '/' + img_name)
					print('path' + '/' + img_name + ': ' + str(np.shape(image)))			