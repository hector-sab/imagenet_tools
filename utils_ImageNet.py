# Validation and bounding boxes
# http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
import os
import cv2
import json
import numpy as np
from random import shuffle

import xml.etree.ElementTree as ET

def bboxes_loader_xml_imagenet(path,args=None):
	# args (tuple): Not needed. If provided:
	#      args[0] (tuple): Shape of the image 
	#         from where the labels where taken.
	#         (height,width)
	#      args[1] (tuple): Shape of the desired 
	#         image  where the labels will be placed.
	#         (height,width)
	# Returns a list of lists with [x_left,y_top,x_right,y_bottom,label]
	
	orig_shape = None
	new_shape = None

	if args is not None:
		orig_shape = args[0]
		new_shape = args[1]

	bboxes = []

	tree = ET.parse(path)
	root = tree.getroot()
	for obj in root.findall('object'):
		# Read the class label
		label = obj.find('name').text

		# Read the bbox
		bbox = obj.find('bndbox')
		x_left = float(bbox.find('xmin').text)
		y_top = float(bbox.find('ymin').text)
		x_right = float(bbox.find('xmax').text)
		y_bottom = float(bbox.find('ymax').text)

		if orig_shape is not None and new_shape is not None:
			x_left = x_left*new_shape[1]/orig_shape[1]
			y_top = y_top*new_shape[0]/orig_shape[0]
			x_right = x_right*new_shape[1]/orig_shape[1]
			y_bottom = y_bottom*new_shape[0]/orig_shape[0]

		bbox = [int(x_left),int(y_top),int(x_right),int(y_bottom),label]
		bboxes.append(bbox)
	return(bboxes)

class ImageNetClassifierData:
	# When you have the bbox location of the objects, and you want to provide
	# the exact portion of the object in the image, use this class.
	def __init__(self,fpaths,ims_dir,bboxes_lbs_dir,json_path,
		rand=True):
		self.ims_dir = ims_dir # Directory of the folders containing the images
		self.bboxes_dir = bboxes_lbs_dir # Directory of the folders containing the xml's

		self.paths = self.load_fpaths(fpaths) # Contains the paths to all the data
		if rand: shuffle(self.paths)

		self.id2class_dict = self.load_json(json_path) # Contains a dict with
		                                              # the conversion between ids
		                                              # and classes {'n04012':['person',14]}

		self.total_items = len(self.paths)
		self.current_ind = 0

	def load_fpaths(self,path):
		# Loads the paths of all the data into a list
		with open(path,'r') as f:
			lines = f.readlines()

		paths = []
		for line in lines:
			line = line.strip('\n')
			line = line.split(' ')
			paths.append(line[0])
		return(paths)

	def load_json(self,path):
		# Loads the dictionary (json file) containing the conversion
		# between id's and class names
		id2class = None
		if path is not None and os.path.exists(path):
			with open(path,'r') as f:
				id2class = json.loads(f.read())
		return(id2class)

	def load_image(self,path):
		im = None
		if os.path.exists(path):
			im = cv2.imread(path)
			im = im[...,::-1]
			im = im.astype(np.uint8)
		return(im)

	def id2class(self,ident):
		clss = None
		if self.id2class_dict is not None:
			if ident in self.id2class_dict.keys():
				clss = self.id2class_dict[ident][1] # 0: Name of the class
				                                    # 1: Number of the class
		return(clss)

	def get_single_object_from_im(self,im,bbox):
		# Returns the area of the image where the object is
		# bbox [x_left,y_top,x_right,y_bottom]
		ob_im = im[bbox[1]:bbox[3],bbox[0]:bbox[2],...]
		return(ob_im)

	def convert2square_im(self,im,shape=(208,208)):
		# shape: (Height,Width)
		h = im.shape[0]
		w = im.shape[1]

		biggest = None
		smallest = None
		if h>w:
			# Creates the container for the square image
			im_sq = np.zeros((h,h,3))
			# Calculates the padding to make it square
			wp = int((h-w)/2)
			# Place the image in the container
			im_sq[:,wp:wp+w] = im
		elif w>h:
			im_sq = np.zeros((w,w,3))
			
			hp = int((w-h)/2)
			im_sq[hp:hp+h,:] = im
		else:
			im_sq = im

		im_sq = im_sq.astype(np.uint8)

		# Makes the image of the desired size
		im_sq = cv2.resize(im_sq,shape)

		return(im_sq)

	def get_single_file(self,im_path,bboxes_path):
		im = self.load_image(im_path)

		ims = [] # Contains the object images
		clss = [] # Cotnains the classes of the objects
		bboxes = bboxes_loader_xml_imagenet(bboxes_path)
		for bbox in bboxes:
			clss.append(self.id2class(bbox[4]))
			bbox_im = self.get_single_object_from_im(im,bbox)
			bbox_im = self.convert2square_im(bbox_im)
			ims.append(bbox_im)

		return(ims,clss)

	def next_batch(self,bs=1):
		# Retrieve a batch of data. When it reaches the last example
		# the lists will be empty.
		# It returns a dictionary {'images':[],'classes':[],'bboxes':[]}
		# 
		# Args:
		#    bs (int): batch size

		ind = self.current_ind
		if ind<self.total_items-1:
			paths = self.paths[ind:ind+bs]
		else:
			paths = []

		ims = []
		classes = []

		for path in paths:
			im_path = os.path.join(self.ims_dir,path+'.JPEG')
			bbox_path = os.path.join(self.bboxes_dir,path+'.xml')

			im,clss = self.get_single_file(im_path,bbox_path)
			ims += im
			classes += clss

		self.current_ind += bs

		return({'images':ims,'classes':classes})

	def reset(self):
		self.current_ind = 0

class ImageNetTool:
	# Object that facilitates the retrieval of data from the ImageNet
	# dataset.
	def __init__(self,fpaths,ims_dir,bboxes_dir=None,json_path=None,
		rand=False,val_lbs_paths=None):
		# Args:
		#    fpaths (str): Path to the txt file containing the paths of the files
		#    ims_dir (str): Directory containing the folders of the images
		#    bboxes_dir (str): Directory containing the folders of the xml files
		#    json_path (str): Path to the json file containing the conversion
		#        between the id's and the name of the classes

		self.paths = self.load_fpaths(fpaths) # Contains the paths to all the data
		
		# For validation...
		self.val_lbs = val_lbs_paths
		if self.val_lbs is not None:
			self.val_lbs = self.load_val_lbs(val_lbs_paths)
			
		if rand and self.val_lbs is None: 
			shuffle(self.paths)
		else:
			tmp = list(zip(self.paths,self.val_lbs))
			shuffle(tmp)
			self.paths,self.val_lbs = zip(*tmp)
		self.ims_dir = ims_dir # Directory of the folders containing the images
		self.bboxes_dir = bboxes_dir # Directory of the folders containing the xml's
		
		self.id2class_dict = self.load_json(json_path) # Contains a dict with
		                                              # the conversion between ids
		                                              # and classes {'n4080':['person',14]}
		
		self.total_items = len(self.paths)
		self.current_ind = 0

	def load_json(self,path):
		# Loads the dictionary (json file) containing the conversion
		# between id's and class names
		id2class = None
		if path is not None and os.path.exists(path):
			with open(path,'r') as f:
				id2class = json.loads(f.read())
		return(id2class)

	def load_fpaths(self,path):
		# Loads the paths of all the data into a list
		with open(path,'r') as f:
			lines = f.readlines()

		paths = []
		for line in lines:
			line = line.strip('\n')
			line = line.split(' ')
			paths.append(line[0])
		return(paths)

	def load_val_lbs(self,path):
		with open(path,'r') as f:
			lines = f.readlines()
		lbs = []
		for line in lines:
			line = line.strip('\n')
			lbs.append(int(line))
		return(lbs)

	def load_image(self,path):
		im = None
		if os.path.exists(path):
			im = cv2.imread(path)
			im = im[...,::-1]
			im = im.astype(np.uint8)
		return(im)

	def id2class(self,ident):
		clss = None
		if self.id2class_dict is not None:
			if ident in self.id2class_dict.keys():
				clss = self.id2class_dict[ident][0] # Name of the class
		return(clss)

	def get_single_data(self,im_path,ident,bboxes_path=None):
		clss = self.id2class(ident) # Name of the class

		im = self.load_image(im_path)

		bboxes = None
		if bboxes_path is not None and os.path.exists(bboxes_path):
			bboxes = bboxes_loader_xml_imagenet(bboxes_path)

		return(im,clss,bboxes)

	def next_batch(self,bs=0):
		# Retrieve a batch of data. When it reaches the last example
		# the lists will be empty.
		# It returns a dictionary {'images':[],'classes':[],'bboxes':[]}
		# 
		# Args:
		#    bs (int): batch size

		ind = self.current_ind
		if ind<self.total_items-1:
			paths = self.paths[ind:ind+bs]
		else:
			paths = []

		ims = []
		classes = []
		bboxes = []

		for i in range(len(paths)):
			path = paths[i]
			im_path = os.path.join(self.ims_dir,path+'.JPEG')
			bbox_path = None
			if self.bboxes_dir is not None:
				bbox_path = os.path.join(self.bboxes_dir,path+'.xml')

			ident = path.split('/')[0]

			im,clss,bbox = self.get_single_data(im_path,ident,bbox_path)

			# If self.val_lbs_paths is not None, ident will be None.
			# To fix it, use val_lbs instead
			if self.val_lbs is not None:
				clss = self.val_lbs[i]

			ims.append(im)
			classes.append(clss)
			bboxes.append(bbox)

		self.current_ind += bs

		return({'images':ims,'classes':classes,'bboxes':bboxes})

	def reset(self):
		self.current_ind = 0

if __name__=='__main__':
	ims_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Data/CLS-LOC/train/'
	bboxes_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Annotations/CLS-LOC/train/'
	fpaths_path = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/ImageSets/CLS-LOC/train_loc.txt'
	json_path = 'id2class.json'

	if False:
		tool = ImageNetTool(fpaths=fpaths_path,ims_dir=ims_dir,bboxes_dir=bboxes_dir,
			json_path=json_path)
	else:
		# It will return the image with the exact location of the object and its class
		tool = ImageNetClassifierData(fpaths=fpaths_path,ims_dir=ims_dir,
			bboxes_lbs_dir=bboxes_dir,json_path=json_path)