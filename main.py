import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
import keras
from scipy import misc
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Config(object):
	def __init__(self):
		self.num_classes = 10
		self.num_epochs = 10
		self.max_seq_len = 22
		self.batch_size = 32
		self.MAX_WIDTH = 90
		self.MAX_HEIGHT = 90

class LipReader(object):
	def __init__(self, config):
		self.config = config		
		#self.config.batch_size = np.shape(self.X_train)[0]
		#self.config.batch_size_val = np.shape(self.X_val)[0]
	
	def create_model(self):
		model = Sequential()

		conv2d = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation='relu')
		timeDistributed = keras.layers.wrappers.TimeDistributed(conv2d, input_shape=(self.config.max_seq_len, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
		model.add(timeDistributed)
		reshape1 = keras.layers.wrappers.TimeDistributed(keras.layers.core.Flatten())
		model.add(reshape1)
		#pool = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
		
		lstm = keras.layers.recurrent.LSTM(512)
		model.add(lstm)

		dense1 = keras.layers.core.Dense(10)
		model.add(dense1)

		activation1 = keras.layers.core.Activation('softmax')
		model.add(activation1)

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		#keras.preprocessing.sequence.pad_sequences(sequences, maxlen=22, padding='pre', value=0.)
		#self.X_train = np.reshape(self.X_train, (self.config.batch_size, -1, 480 * 640 * 3))
		#self.X_val = np.reshape(self.X_val, (self.config.batch_size, -1, 480 * 640 * 3))
		#y_train doesnt have to be reshaped -- already is an array of size batch_length

		one_hot_labels_train = keras.utils.to_categorical(self.y_train, num_classes=self.config.num_classes)
		one_hot_labels_val = keras.utils.to_categorical(self.y_val, num_classes=self.config.num_classes)
		
		print('Fitting the model...')
		history = model.fit(self.X_train, one_hot_labels_train, epochs=self.config.num_epochs, batch_size=self.config.batch_size,\
							validation_data=(self.X_val, one_hot_labels_val))

		self.create_plots(history)

		#keras.utils.plot_model(model, to_file='model.png')

		'''
		print('Evaluating the model...')
		score = model.evaluate(self.X_val, one_hot_labels_val, batch_size=self.config.batch_size)

		print('Finished training, with the following val score:')
		print(score)
		'''

	'''
	def create_minibatches(self, data, shape):
		data = [self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test]
		for dataset in 	
			batches = []
			for i in range(0, len(data), self.config.batch_size)
				sample = data[i:i + self.config.batch_size]
				if len(sample) < self.config.batch_size:
					pad = np.zeros(shape)
					sample.extend(pad * (size - len(sample)))
				batches.append(sample)
	'''

	def create_plots(self, history):
		os.mkdir('plots')
		# summarize history for accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig('plots/acc_plot.png')
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig('plots/loss_plot.png')


	def load_data(self):

		if os.path.exists('data'):
			print('loading saved data...')
			self.X_train = np.load('data/X_train.npy')
			self.y_train = np.load('data/y_train.npy')

			self.X_val = np.load('data/X_val.npy')
			self.y_val = np.load('data/y_val.npy')

			self.X_test = np.load('data/X_test.npy')
			self.y_test = np.load('data/y_test.npy')
			print('Read data arrays from disk.npy')
			
			#self.X_test = np.reshape(self.X_test, (np.shape(self.X_test)[0], -1, 480*640*3))


		else:

			people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
			#people = ['F01','F02','F04','F05', 'F06']
			
			#removed 'phrases' temporarily from data types
			data_types = ['words']
			
			folder_enum = ['01','02','03','04','05','06','07','08','09','10']
			#folder_enum = ['01','02','03','04', '05']

			VALIDATION_SPLIT = ['F05']
			TEST_SPLIT = ['F06']

			self.X_train = []
			self.y_train = []

			self.X_val = []
			self.y_val = []

			self.X_test = []
			self.y_test = [] 

			directory = '../../cropped'
			for person_id in people:
				for word_index, word in enumerate(folder_enum):
					for iteration in folder_enum:
						path = os.path.join(directory, person_id, 'words', word, iteration)
						filelist = os.listdir(path + '/')
						sequence = []
						for img_name in filelist:
							if img_name.startswith('color'):
								image = misc.imread(path + '/' + img_name)
								image = image[:self.config.MAX_WIDTH,:self.config.MAX_HEIGHT,...]
								#image = np.reshape(image, self.config.MAX_WIDTH*self.config.MAX_HEIGHT*3)
								sequence.append(image)
								print("read: " + path + '/' + img_name)
						pad_array = [np.zeros((self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))]
						sequence.extend(pad_array * (self.config.max_seq_len - len(sequence)))
						sequence = np.stack(sequence, axis=0)
						if person_id in TEST_SPLIT:
							self.X_test.append(sequence)
							self.y_test.append(word_index)
						elif person_id in VALIDATION_SPLIT:
							self.X_val.append(sequence)
							self.y_val.append(word_index)
						else:
							self.X_train.append(sequence)
							self.y_train.append(word_index)
				print('Finished reading images for person ' + person_id)
			
			print('Finished reading images.')
			self.X_train = np.stack(self.X_train, axis=0)	
			self.X_val = np.stack(self.X_val, axis=0)
			self.X_test = np.stack(self.X_test, axis=0)
			print('Finished stacking the data into the right dimensions. About to start saving to disk...')		
			os.mkdir('data')
			np.save('data/X_train', self.X_train)
			np.save('data/y_train', np.array(self.y_train))
			np.save('data/X_val', self.X_val)
			np.save('data/y_val', np.array(self.y_val))
			np.save('data/X_test', self.X_test)
			np.save('data/y_test', np.array(self.y_test))
			print('Finished saving all data to disk.')

		print('X_train shape: ', np.shape(self.X_train))
		print('y_train shape: ', np.shape(self.y_train))

		print('X_val shape: ', np.shape(self.X_val))
		print('y_val shape: ', np.shape(self.y_val))

		print('X_test shape: ', np.shape(self.X_test))
		print('y_test shape: ', np.shape(self.y_test))

if __name__ == '__main__':
	config = Config()
	lipReader = LipReader(config)
	lipReader.load_data()
	lipReader.create_model()
