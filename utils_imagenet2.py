import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from random import shuffle
import xml.etree.ElementTree as ET

def load_image(path):
		im = None
		if os.path.exists(path):
			im = cv2.imread(path)
			# From BGR to RGB
			im = im[...,::-1]
			im = im.astype(np.uint8)
		return(im)

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

def _bytes_feature(value):
	# Used for images
	return(tf.train.Feature(bytes_list=tf.train.BytesList(
		value=[value])))

def _int64_feature(value):
		# Used for int
		return(tf.train.Feature(int64_list=tf.train.Int64List(
			value=[value])))

class ILSVRC2012TrainDataLoader:
	def __init__(self,fpaths,ims_dir,lbs_dir=None,
		json_path=None,rand=False):
		# Args:
		#    fpaths (str): Path to the txt file containing the paths of the files
		#    ims_dir (str): Directory containing the folders of the images
		#    bboxes_dir (str): Directory containing the folders of the xml files
		#    json_path (str): Path to the json file containing the conversion
		#        between the id's and the name of the classes

		self.paths = self.load_fpaths(fpaths) # Contains the paths to all the data
		if rand: shuffle(self.paths)
		self.ims_dir = ims_dir # Directory of the folders containing the images
		self.lbs_dir = lbs_dir # Directory of the folders containing the xml's

		self.id2class_dict = self.load_json(json_path) # Contains a dict with
		                                              # the conversion between ids
		                                              # and classes
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

	def id2class(self,ident):
		# Converts the id 
		clss = None
		if self.id2class_dict is not None:
			if ident in self.id2class_dict.keys():
				clss = self.id2class_dict[ident][1] # 0: Name of the class
				                                    # 1: class ind
		return(clss)

	def get_single_data(self,im_path,ident,bbox_path=None):
		clss = self.id2class(ident) # Name of the class
		im = load_image(im_path)

		bboxes = None
		if bbox_path is not None and os.path.exists(bbox_path):
			bbox = bboxes_loader_xml_imagenet(bbox_path)

		return(im,clss,bboxes)

	def next_batch(self,bs=1):
		# Retrieve a batch of data. When it reaches the last example
		# the lists will be empty.
		# It returns a dictionary {'images':[],'classes':[],'bboxes':[]}
		# 
		# Args:
		#    bs (int): batch size
		ind = self.current_ind

		paths = []
		if ind<self.total_items-1:
			paths = self.paths[ind:ind+bs]

		ims = []
		classes = []
		bboxes = []

		for path in paths:
			im_path = os.path.join(self.ims_dir,path+'.JPEG')

			bbox_path = None
			if self.lbs_dir is not None:
				bbox_path = os.path.join(self.lbs_dir,path+'.xml')
			ident = path.split('/')[0] # Identifier

			im,clss,bbox = self.get_single_data(im_path,ident,bbox_path)

			ims.append(im)
			classes.append(clss)
			bboxes.append(bbox)

		return({'images':ims,'classes':classes,'bboxes':bboxes})

	def reset(self):
		self.current_ind = 0

def ILSVRC2012TrainCLSDataWriter(batch,writer):
	# https://mc.ai/storage-efficient-tfrecord-for-images/
	for im,clss in zip(batch['images'],batch['classes']):
		encoded_im_str = cv2.imencode('.jpg',im)[1].tostring()
		
		im = tf.compat.as_bytes(encoded_im_str)
		#clss = tf.compat.as_bytes(clss)
		
		feature = {
			'image':_bytes_feature(im),
			'height':_int64_feature(im.shape[0]),
			'width':_int64_feature(im.shape[1]),
			'label':_int64_feature(clss)
		}

		tf_example = tf.train.Example(
			features=tf.train.Features(feature=feature))
		writer.write(tf_example.SerializeToString())

class TFRecordsCreator:
	def __init__(self,data_loader,data_writer):
		self.data_loader = data_loader
		self.data_writer = data_writer
		self.total_items = self.data_loader.total_items
		self.idx_tfrecord = 0 # Used to name tfrecord files

	def create(self,bs=10,out_dir='./tfrecords/',init_tfind=None):
		# Makes sure the output dir exists
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		# Initialize the naminf of the files
		if init_tfind is not None:
			self.idx_tfrecord = init_tfind

		# Claculate the required number of iterations
		total_its = int(self.total_items/bs)
		if self.total_items%bs!=0:
			total_its += 1

		for _ in tqdm(range(total_its)):
			batch = self.data_loader.next_batch(bs)
			fname = str(self.idx_tfrecord)+'.tfrecords'
			filename_tfrecord = os.path.join(out_dir,fname)
			with tf.python_io.TFRecordWriter(filename_tfrecord) as writer:
				self.data_writer(batch,writer)
			self.idx_tfrecord += 1




def _extract_features(example):
	features = {
			'image':tf.FixedLenFeature([],tf.string),
			'label':tf.FixedLenFeature([], tf.int64)
		}
	parse_example = tf.parse_single_example(example,features)
	image = tf.decode_raw(parsed_example['image'],tf.float32)
	label = tf.cast(parsed_example['image'],tf.int32)
	return(image,label)


class ILSVRC2012TrainTFRecordDataLoader:
	def __init__(self,data_dir):
		#https://medium.com/@dikatok/making-life-better-read-easier-with-tensorflow-dataset-api-fb91851e51f4
		fnames = sorted(os.listdir(data_dir))
		fnames = [os.path.join(data_dir,x) for x in self.fnames]
		self.dataset = tf.data.TFRecordDataset(self.fnames)

		
		#http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
		self.features = {
			'image':tf.FixedLenFeature([],tf.string),
			'label':tf.FixedLenFeature([], tf.int64)
		}

		

	def next_batch(self):
		features = tf.parse_single_example()
		

class TFRecordsLoader:
	def __init__(self,data_loader,data_dir):
		self.data_loader = data_loader
		self.data_dir = data_dir

if __name__=='__main__':
	# Images dir
	ims_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Data/CLS-LOC/train/'
	# Bbox labels dir
	lbs_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Annotations/CLS-LOC/train/'
	# File with the relative paths of the data files
	fpaths_path = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/ImageSets/CLS-LOC/train_loc.txt'
	json_path = 'id2class.json'

	data_loader = ILSVRC2012TrainDataLoader(fpaths_path,ims_dir,lbs_dir,json_path,rand=True)
	data_writer = ILSVRC2012TrainCLSDataWriter
	creator = TFRecordsCreator(data_loader,data_writer)

	creator.create(out_dir='/data/HectorSanchez/database/imagenet_tfrecords/train_cls/')