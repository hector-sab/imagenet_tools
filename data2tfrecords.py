import os
from tqdm import tqdm
import tensorflow as tf

from utils_ImageNet import ImageNetTool

def convert2square_im(im,shape=(208,208)):
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

class TFRCreatorClassifier:
	def __init__(self):
		self.ims_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Data/CLS-LOC/train/'
		self.bboxes_dir = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/Annotations/CLS-LOC/train/'
		self.fpaths_path = '/data/DataBases/ImageNet/Object_Localization/ILSVRC/ImageSets/CLS-LOC/train_loc.txt'
		self.json_path = 'id2class.json'

		self.loader = ImageNetTool(fpaths=fpaths_path,ims_dir=ims_dir,bboxes_dir=bboxes_dir,
			json_path=json_path)

		self.total_items = self.loader.total_items

		self.idx_tfrecord = 0

	# conversion functions (data to feature data types)
	def _bytes_feature(self,value):
		return(tf.train.Feature(bytes_list=tf.train.BytesList(
			value=[value])))

	def data2tfrecord(self,im,ind,writer):
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'image': self._bytes_feature(im.tostring()),
					'label': self._bytes_feature(ind.tostring())
				}))
		writer.write(example.SerializeToString())

	def run(self,bs=10,out_dir='./tfrecords/',init_tfind=None):
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		if init_tfind is not None:
			self.idx_tfrecord = init_tfind

		total_its = (self.total_items/bs)
		if self.total_items%bs!=0:
			total_its += 1
		
		for i in tqdm(range(total_its)):
			batch = self.loader.next_batch(bs)
			filename_tfrecord = os.path.join(out_dir,str(self.idx_tfrecord)+'.tfrecords')
			
			with tf.python_io.TFRecordWriter(filename_tfrecord) as writer:
				for im,class_name in zip(batch['images'],batch['classes']):
					im = convert2square_im(im)
					ind = tool.id2class_dict[class_name][1]
					self.data2tfrecord(im,ind,writer)

			self.idx_tfrecord += 1



		

if __name__=='__main__':

	tool = TFRCreatorClassifier()
	tool.run()


