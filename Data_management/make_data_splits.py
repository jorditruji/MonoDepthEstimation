# -*- coding: utf-8 -*-

from glob import glob
from random import shuffle
import numpy as np
import argparse

'One use script to create the dataset partitions and store them in dataset.npy'



def read_scenes(path):
	"""
	Reads all the dirs(scenes) for a given dataset path organized by scenes.


	Args:
		path (str): Path to dataset directory.


	Returns:
		paths (list): Each path corresponds to a scene
	"""
	return glob(path+'*/')


def read_files(path, extension):
	"""
	Read all files that match a given extenstion in a given path.


	Args:
		path (str): Path to dataset directory.
		extension (str): Extension of the files to read. Ex: mp3, pickle, json...


	Returns:
		filepaths (list):  Path to the files found


	"""
	return glob(path+'*.'+extension)


def make_partitions(scenes, output_filename, sample_extension, n_folds=1, train_size = 0.7, val_size = 0.2, test_size = 0.1):
	"""
	Splits the dataset in 3 partitions at scene level to avoid data leaks between the partitions.
	Those partitions are stored at disk as a dict with train, val and test as keys which contain
	the paths to the images for each partition


	Args:
		scenes (list): List of paths to the dataset scenes
		n_folds (int): Amount of folds 
		train_size (int): Proportion of train samples
		val_size (int): Proportion of val samples
		test_size (int): Proportion of test samples


	"""
	assert train_size+val_size+test_size == 1
	n_scenes = len(scenes)
	# Initialize dict to store data paths
	dataset = {}
	dataset['train']=[]
	dataset['val']=[]
	dataset['test']=[]
	print("Found {}  scenes".format(n_scenes))
	for _i in range(n_partitions):
		shuffle(scenes)
		train = scenes[0:int(0.6*n_scenes)]
		val_test = scenes[int(0.6*n_scenes):]
		val =  val_test[0:int(0.5*len(val_test))]
		test =  val_test[int(0.5*len(val_test)):]
		print("{} training scenes\n{} validation scenes\n{} testing scenes".format(len(train), len(val), len(test)))
		
		# Read depth file names and add them to dataset splits
		for _j, scene in enumerate(train):
			dataset['train']+=read_files(scene, sample_extension)
		print("Train samples {}".format(len(dataset['train'])))

		for _j, scene in enumerate(val):
			dataset['val']+=read_files(scene, sample_extension)
		print("Validation samples {}".format(len(dataset['val'])))

		for _j, scene in enumerate(test,):
			dataset['test']+=read_files(scene, sample_extension)
		print("Test samples {}".format(len(dataset['test'])))

		print("Saving data!!!! Finished")
		np.save(output_filename+str(_i),dataset)

if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", default=None, type=str, required=True,
	                    help="The dataset to process path. Is should contain the folders corresponding to each scene.")

	parser.add_argument("--data_extension", default=None, type=str, required=False,
	                    help="Extension of the samples, should only match inputs or ground_truths but not both.")

	parser.add_argument("--output_filename", default=None, type=str, required=False,
	                    help="Name of the file to save the partitions.")

	args = parser.parse_args()

	DATA_PATH = args.dataset_path
	if DATA_PATH[-1]!= '/':
		DATA_PATH= DATA_PATH+'/'

	output_filename = args.output_filename
	if output_filename is None:
		output_filename = DATA_PATH.split('/')[-1]


	data_extension = args.data_extension

	scenes = read_scenes(DATA_PATH)
	make_partitions(scenes, output_filename, data_extension)
