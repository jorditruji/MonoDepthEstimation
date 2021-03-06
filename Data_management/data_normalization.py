from dataset import NYUDataset
import numpy as np
from torch.utils import data
import argparse
from tqdm import tqdm

if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default=None, type=str, required=True,
	                    help="Options: NYUDataset, ScanNet.")

	parser.add_argument("--output_filename", default=None, type=str, required=True,
	                    help="Extension of the samples, should only match inputs or ground_truths but not both.")

	parser.add_argument("--dataset_partitions", default=None, type=str, required=True,
	                    help="Extension of the samples, should only match inputs or ground_truths but not both.")


	args = parser.parse_args()


	# Load train partition:
	train_depth = np.load(args.dataset_partitions, allow_pickle = True).item()['train']


	# Create dataset instance and dataloader
	# Select dataset loader
	if args.dataset.lower() == 'nyudataset':
		dataset =NYUDataset(train_depth, is_train = False)


	means = []
	stds = []
	vec_maxs = []
	vec_mins = []
	depth_bins = set()
	for _idx in tqdm(range(len(train_depth))):
		# Read and convert to numpies
		depth, rgb, path = dataset.__getitem__(_idx)
		depth, rgb = np.array(depth), np.array(rgb,dtype = float)/255.
		#print(depth.shape, rgb.shape)
		depth_bins.update(set(list(depth.flatten())))
		# Calculate interesting data
		mean = np.mean(rgb, axis = (0,1))
		std =  np.std(rgb, axis = (0,1))

		max_depth = np.max(depth)
		min_depth = np.min(depth)

		# Append data
		means.append(mean)
		stds.append(std)
		vec_mins.append(min_depth)
		vec_maxs.append(max_depth)
		#print("Mean: {} \n STD: {} \n max_depth: {} \n min_depth: {}".format(mean, std, max_depth, min_depth))
		#print(len(depth_bins))
	# Calculate interesting data
	mean = np.mean(means, axis = 0)
	std =  np.mean(stds, axis = 0)
	np.save('means.npy', mean)
	np.save('stds.npy', std)
	print(mean)
	print(std)
	np.save('max_depths.npy',vec_maxs)
	np.save('min_depths.npy', vec_mins)
	print(len(depth_bins))
