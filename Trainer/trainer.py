import torch






class Trainer:
	"""
	Reads and augments the dataset images according to its tate and parameters. Base generic class implementing the common methods.

	:ivar metrics (dict): List of the depth images paths 
	:ivar epoch (boolean): Load images for train/inference 
	:ivar model (albumentation or str): Loads augmentator config from path if str and sets it to attr transforms
	:ivar state_dict (dict): List of the depth images paths 
	:ivar optimizer (boolean): Load images for train/inference 
	:ivar optimizer_dict (albumentation or str): Loads augmentator config from path if str and sets it to attr transforms
	"""
	def __init__(self, training_generator, 
					val_generator,
					model, 
					optimizer)
		self.model = model
		self.optimizer = optimizer
		self.training_generator  = training_generator
		self.val_generator = val_generator


	def load_checkopint(self, filename):
		checkpoint = torch.load(filename)
		self.model = checkpoint['model']
		self.model.load_state_dict(checkpoint['state_dict'])
		self.optimizer = checkpoint['optimizer']
		self.optimizer.load_state_dict(checkpoint['optimizer_dict']) 	

	@classmethod
	def init_metrics(cls, metrics):
		metrics = {phase: [] for phase in ("train", "val")}
		metrics["train"] = {metric:list() for metric in metrics}
		metrics["val"] = {metric:list() for metric in metrics}
		return metrics

	def save_checkpoint(self):
		checkpoint = {'model': model_class,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}


