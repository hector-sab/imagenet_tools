# ImageNet Tools

In here you will find an object that facilitates the retrival of data from the ImageNet dataset.

## How it works

First of all, we need to define: `fpaths_path`, the path of the text file containing the paths of the images/xml's files on the dataset; `ims_dir`, the directory that contains all the folders that contain all the images; `bboxes_dir`, the directory that contains all the folders that contain all the xml files; `json_path`, the json file that contains the mapping between id's and the actual class name.

```python
fpaths_path = 'ILSVRC/ImageSets/CLS-LOC/train_cls.txt'
ims_dir = 'ILSVRC/Data/CLS-LOC/train/'
bboxes_dir = 'ILSVRC/Annotations/CLS-LOC/train/'
json_path = 'id2class.json'
```

Next, import the object class `ImageNetTool`, initialize an object, and ask for the data.

```python
from utils_ImageNet.py import ImageNetTool

# Initialize it
tool = ImageNetTool(fpaths=fpaths_path,ims_dir=ims_dir,bboxes_dir=bboxes_dir,
		json_path=json_path)

# Ask for the data and the size of the batch
data = tool.get_data(bs=2)
```

`data` is a dictionary containing three elements: `'images'`,`'classes'`,`'bboxes'`. Each element is a list containing the images (cv2|np), classes (str), and bboxes (list).

`data['bboxes']` is a list of list with the form `[[x_min,y_min,x_max,y_max,class]]`.

**NOTE:** `readable_id_classes.txt` was taken from [here.](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57)