from dataset import NYUDataset
import numpy as np


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
	data_partitions = np.load()
	if args.dataset.lower() == 'nyudataset':
		dataset =NYUDataset()
