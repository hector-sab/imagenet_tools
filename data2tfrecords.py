import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils_ImageNet import ImageNetTool

def convert2square_im(im,shape=None):
	# Makes an image square by adding padding in the
	# sides that need it. If shape is given, reshapes the
	# image.
	# Args:
	#     shape (tuple): (Height,Width)
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
	if shape is not None:
		im_sq = cv2.resize(im_sq,shape)

	return(im_sq)

class TFRCreatorImageNetClassifier:
	def __init__(self,fpaths_path,ims_dir,lbs_dir,json_path):
		# fpaths_path (str): file containing the paths of the data
		# ims_dir (str): Directory where the images are located
		# lbs_dir (str): Directory where the labels are located
		# json_path (str): Path to a json file used to convert from ids
		#      to class name or ind.

		self.fpaths_path = fpaths_path
		self.ims_dir = ims_dir
		self.lbs_dir = lbs_dir
		self.json_path = json_path


		self.data = ImageNetTool(fpaths=self.fpaths_path,ims_dir=self.ims_dir,
			bboxes_dir=self.bboxes_dir,json_path=self.json_path)

		self.total_items = self.data.total_items
		self.idx_tfrecord = 0 # Used to name tfrecord files

		# Dictionary to convert class names to indexes
		self.clss2ind = {}
		for key in self.data.id2class_dict:
			cname = self.data.id2class_dict[key][0]
			self.clss2ind[cname] = self.loader.id2class_dict[key][1]

		self.preprocessing = None

	def _bytes_feature(self,value):
		# Used for images
		return(tf.train.Feature(bytes_list=tf.train.BytesList(
			value=[value])))

	def _int64_feature(self,value):
		# Used for int
		return(tf.train.Feature(int64_list=tf.train.Int64List(
			value=[value])))

	def data2tfrecord(self,im,ind,writer):
		# im (np.array): Image to save
		# ind (int): Class index of the image
		# writer: Writer
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'image': self._bytes_feature(im.tostring()),
					'label': self._int64_feature(ind)
				}))
		writer.write(example.SerializeToString())

	def set_preprocessing(self,function):
		# Set the preprocessing of the images
		# before they are stored
		self.preprocessing = function

	def run(self,bs=10,out_dir='./tfrecords/',init_tfind=None):
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

		for i in tqdm(range(total_its)):
			batch = self.data.next_batch(bs)
			filename_tfrecord = os.path.join(out_dir,str(self.idx_tfrecord)+'.tfrecords')
			
			with tf.python_io.TFRecordWriter(filename_tfrecord) as writer:
				for im,class_name in zip(batch['images'],batch['classes']):
					if preprocessing is not None
						im = self.preprocessing(im)
					ind = self.clss2ind[class_name]
					self.data2tfrecord(im,ind,writer)

			self.idx_tfrecord += 1

class TFRCreatorClassifier:
	def __init__(self):
		self.ims_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Data/CLS-LOC/train/'
		self.bboxes_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Annotations/CLS-LOC/train/'
		self.fpaths_path = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/ImageSets/CLS-LOC/train_loc.txt'
		self.json_path = 'id2class.json'

		self.loader = ImageNetTool(fpaths=self.fpaths_path,ims_dir=self.ims_dir,
			bboxes_dir=self.bboxes_dir,json_path=self.json_path)

		self.total_items = self.loader.total_items

		self.idx_tfrecord = 0

		self.clss2ind = {}
		for key in self.loader.id2class_dict:
			cname = self.loader.id2class_dict[key][0]
			self.clss2ind[cname] = self.loader.id2class_dict[key][1]

	# conversion functions (data to feature data types)
	def _bytes_feature(self,value):
		return(tf.train.Feature(bytes_list=tf.train.BytesList(
			value=[value])))

	def _int64_feature(self,value):
		return(tf.train.Feature(int64_list=tf.train.Int64List(
			value=[value])))

	def data2tfrecord(self,im,ind,writer):
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'image': self._bytes_feature(im.tostring()),
					'label': self._int64_feature(ind)
				}))
		writer.write(example.SerializeToString())

	def run(self,bs=10,out_dir='./tfrecords/',init_tfind=None):
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		if init_tfind is not None:
			self.idx_tfrecord = init_tfind

		total_its = int(self.total_items/bs)
		if self.total_items%bs!=0:
			total_its += 1
		
		for i in tqdm(range(total_its)):
			batch = self.loader.next_batch(bs)
			filename_tfrecord = os.path.join(out_dir,str(self.idx_tfrecord)+'.tfrecords')
			
			with tf.python_io.TFRecordWriter(filename_tfrecord) as writer:
				for im,class_name in zip(batch['images'],batch['classes']):
					im = convert2square_im(im)
					ind = self.clss2ind[class_name]
					self.data2tfrecord(im,ind,writer)

			self.idx_tfrecord += 1



		

if __name__=='__main__':
	if False:
		tool = TFRCreatorClassifier()
		tool.run(out_dir='/data/HectorSanchez/database/imagenet_tfrecords/train_cls/')
	else:
		ims_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Data/CLS-LOC/train/'
		lbs_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Annotations/CLS-LOC/train/'
		fpaths_path = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/ImageSets/CLS-LOC/train_loc.txt'
		json_path = 'id2class.json'

		tool = TFRCreatorImageNetClassifier(fpaths_path=fpaths_path,ims_dir=ims_dir,
			lbs_dir=lbs_dir,json_path=json_path)
		
		tool.run(out_dir='/data/HectorSanchez/database/imagenet_tfrecords/train_cls/')
