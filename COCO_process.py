#sno 2
from include import *

def data_process_COCO(annotation_file, PATH):


	# This function returns image-caption pairs with 5 captions per image

	with open(annotation_file, 'r') as f:
		anno = json.load(f)


	capTrain = []
	imgNam = []
	img_cap = collections.defaultdict(list)
	
	va = 1
	for node in anno['annotations']:
		caption = f"<start> {node['caption']} <end>"
		imgpath = PATH + 'COCO_train2014_' + '%012d.jpg' % (node['image_id'])
		if va ==1:
			print(imgpath)
			va=0
		img_cap[imgpath].append(caption)
  


	img_path = list(img_cap.keys())
	train_imgpath = img_path
	print(len(train_imgpath))


	for img_path in train_imgpath:
		capTrain.extend(img_cap[img_path])
		imgNam.extend([img_path] * len(img_cap[img_path]))

	#print("Length of the caption & images vectors")
	#print(len(capTrain))
	#print("----")
	#print(len(imgNam))
	return capTrain, imgNam
