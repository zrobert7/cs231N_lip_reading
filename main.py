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
import argparse
from keras.layers.wrappers import TimeDistributed
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools


class Config(object):
	def __init__(self, nc, ne, msl, bs, lr, dp):
		self.num_classes = nc
		self.num_epochs = ne
		self.max_seq_len = msl
		self.batch_size = bs
		self.learning_rate = lr
		self.MAX_WIDTH = 90
		self.MAX_HEIGHT = 90
		self.dropout = dp
		np.random.seed(1337)

class LipReader(object):
	def __init__(self, config):
		self.config = config		
		#self.config.batch_size = np.shape(self.X_train)[0]
		#self.config.batch_size_val = np.shape(self.X_val)[0]

	def training_generator(self):
		while True:
			for i in range(int(np.shape(self.X_train)[0] / self.config.batch_size)):
				x = self.X_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
				y = self.y_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
				one_hot_labels_train = keras.utils.to_categorical(y, num_classes=self.config.num_classes)
				yield (x,one_hot_labels_train)

	
	def create_model(self, seen_validation):
		np.random.seed(0)
		bottleneck_train_path = 'bottleneck_features_train.npy'
		bottleneck_val_path = 'bottleneck_features_val.npy'
		
		if seen_validation is False:
			bottleneck_train_path = 'unseen_bottleneck_features_train.npy'
			bottleneck_val_path = 'unseen_bottleneck_features_val.npy'

		input_layer = keras.layers.Input(shape=(self.config.max_seq_len, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
				
		vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))

		vgg = Model(input=vgg_base.input, output=vgg_base.output)
		x = TimeDistributed(vgg)(input_layer)

		bottleneck_model = Model(input=input_layer, output=x)


		if not os.path.exists(bottleneck_train_path):
			bottleneck_features_train = bottleneck_model.predict_generator(self.training_generator(), steps=np.shape(self.X_train)[0] / self.config.batch_size)
			np.save(bottleneck_train_path, bottleneck_features_train)

		if not os.path.exists(bottleneck_val_path):
			bottleneck_features_val = bottleneck_model.predict(self.X_val)
			np.save(bottleneck_val_path, bottleneck_features_val)

		#vgg = Model(input=vgg_base.input, output=vgg_base.output)
		#vgg.trainable = False

		#x = TimeDistributed(vgg)(input_layer)


		'''
		conv2d1 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
		x = TimeDistributed(conv2d1)(x) #input_shape=(self.config.max_seq_len, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3)

		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
		x = keras.layers.core.Activation('relu')(x)
		x = keras.layers.core.Dropout(rate=dp)(x)
		
		pool1 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
		x = TimeDistributed(pool1)(x)

		conv2d2 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
		x = TimeDistributed(conv2d2)(x)

		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
		x = keras.layers.core.Activation('relu')(x)
		x = keras.layers.core.Dropout(rate=dp)(x)

		pool2 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
		x = TimeDistributed(pool2)(x)

		conv2d3 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
		x = TimeDistributed(conv2d3)(x)

		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
		x = keras.layers.core.Activation('relu')(x)
		x = keras.layers.core.Dropout(rate=dp)(x)

		pool3 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
		x = TimeDistributed(pool3)(x)
		'''
		train_data = np.load(bottleneck_train_path)
		val_data = np.load(bottleneck_val_path)


		model = Sequential()
		model.add(TimeDistributed(keras.layers.core.Flatten(),input_shape=train_data.shape[1:]))
		
	
		lstm = keras.layers.recurrent.LSTM(256)
		model.add(keras.layers.wrappers.Bidirectional(lstm, merge_mode='concat', weights=None))

		#model.add(keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001))
		#model.add(keras.layers.core.Activation('relu'))
		model.add(keras.layers.core.Dropout(rate=self.config.dropout))

		model.add(keras.layers.core.Dense(10))

		model.add(keras.layers.core.Activation('softmax'))

		#model = Model(inputs=input_layer, outputs=predictions)

		
		adam = keras.optimizers.Adam(lr=self.config.learning_rate)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

		one_hot_labels_train = keras.utils.to_categorical(self.y_train, num_classes=self.config.num_classes)
		one_hot_labels_val = keras.utils.to_categorical(self.y_val, num_classes=self.config.num_classes)
		
		print('Fitting the model...')

		history = model.fit(train_data, one_hot_labels_train, epochs=self.config.num_epochs, batch_size=self.config.batch_size,\
							validation_data=(val_data, one_hot_labels_val))

		predictions = model.predict(bottleneck_model.predict(self.X_test)) 

		labels = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']
		cm = confusion_matrix(self.y_test, predictions, labels=labels)
		plot_confusion_matrix(cm, labels,normalize=False,title='Confusion matrix', cmap=plt.cm.Blues)

		'''
		history = model.fit_generator(self.training_generator(), steps_per_epoch=np.shape(self.X_train)[0] / self.config.batch_size,\
					 epochs=self.config.num_epochs, validation_data=(self.X_val, one_hot_labels_val))
		'''
		#self.create_plots(history)

		#print('Layer names and layer indices:')
		#for i, layer in enumerate(base_model.layers):
			#print(i, layer.name)

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
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig('plots/acc_plot.png')
		plt.clf()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig('plots/loss_plot.png')

	def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig('confusion.png')


	def load_data(self, seen_validation):
		data_dir = 'data'
		if seen_validation:
			data_dir = 'data_seen'

		if os.path.exists('../../' + data_dir):
			print('loading saved data...')
			self.X_train = np.load('../../' +  data_dir + '/X_train.npy')
			self.y_train = np.load('../../'+ data_dir +'/y_train.npy')

			self.X_val = np.load('../../'+ data_dir +'/X_val.npy')
			self.y_val = np.load('../../'+data_dir+'/y_val.npy')

			self.X_test = np.load('../../'+data_dir+'/X_test.npy')
			self.y_test = np.load('../../'+data_dir+'/y_test.npy')
			print('Read data arrays from disk.npy')
			
			#self.X_test = np.reshape(self.X_test, (np.shape(self.X_test)[0], -1, 480*640*3))


		else:

			people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
			#people = ['F01','F02','F04','F05', 'F06']
			
			#removed 'phrases' temporarily from data types
			data_types = ['words']#, 'words_jitter']#, 'words_flip_xaxis']
			
			folder_enum = ['01','02','03','04','05','06','07','08','09','10']
			#folder_enum = ['01','02','03','04']

			UNSEEN_VALIDATION_SPLIT = ['F05']
			UNSEEN_TEST_SPLIT = ['F06']

			SEEN_VALIDATION_SPLIT = ['02']
			SEEN_TEST_SPLIT = ['01']

			self.X_train = []
			self.y_train = []

			self.X_val = []
			self.y_val = []

			self.X_test = []
			self.y_test = [] 

			directory = '../../cropped'
			for person_id in people:
				for data_type in data_types: 
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
							
							if seen_validation == False:
								if person_id in UNSEEN_TEST_SPLIT:
									self.X_test.append(sequence)
									self.y_test.append(word_index)
								elif person_id in UNSEEN_VALIDATION_SPLIT:
									self.X_val.append(sequence)
									self.y_val.append(word_index)
								else:
									self.X_train.append(sequence)
									self.y_train.append(word_index)
							else:
								if iteration in SEEN_TEST_SPLIT:
									self.X_test.append(sequence)
									self.y_test.append(word_index)
								elif iteration in SEEN_VALIDATION_SPLIT:
									self.X_val.append(sequence)
									self.y_val.append(word_index)
								else:
									self.X_train.append(sequence)
									self.y_train.append(word_index)

				print('Finished reading images for person ' + person_id)
			
			print('Finished reading images.')
			print(np.shape(self.X_train))
			self.X_train = np.stack(self.X_train, axis=0)	
			self.X_val = np.stack(self.X_val, axis=0)
			self.X_test = np.stack(self.X_test, axis=0)
			print('Finished stacking the data into the right dimensions. About to start saving to disk...')		
			os.mkdir('../' + data_dir)
			np.save('../'+data_dir+'/X_train', self.X_train)
			np.save('../'+data_dir+'/y_train', np.array(self.y_train))
			np.save('../'+data_dir+'/X_val', self.X_val)
			np.save('../'+data_dir+'/y_val', np.array(self.y_val))
			np.save('../'+data_dir+'/X_test', self.X_test)
			np.save('../'+data_dir+'/y_test', np.array(self.y_test))
			print('Finished saving all data to disk.')

		print('X_train shape: ', np.shape(self.X_train))
		print('y_train shape: ', np.shape(self.y_train))

		print('X_val shape: ', np.shape(self.X_val))
		print('y_val shape: ', np.shape(self.y_val))

		print('X_test shape: ', np.shape(self.X_test))
		print('y_test shape: ', np.shape(self.y_test))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Lip reading model')
	parser.add_argument('--seen_validation', dest='seen_validation', action='store_true')
	#parser.add_argument('--unseen_validation', dest='seen_validation', action='store_false')
	parser.set_defaults(seen_validation=False)
	ARGS = parser.parse_args()
	print("Seen validation: %r" % (ARGS.seen_validation))
	num_epochs = [5]#10
	learning_rates = [0.00006]#, 0.0005]
	batch_size = [50]
	dropout_ = [0.2]
	for ne in num_epochs:
		for bs in batch_size: 
			for lr in learning_rates:
				for dp in dropout_:
					#print 'Epochs: %d    Batch Size: %d Learning Rate: %f  Dropout: %f'%( ne, bs, lr, dp)
					config = Config(10, ne, 22, bs, lr, dp)
					lipReader = LipReader(config)
					lipReader.load_data(ARGS.seen_validation)
					lipReader.create_model(ARGS.seen_validation)
