#sno 1
from include import *

def download_COCO():

  os.chdir("./")
  annotation_folder = 'annotations/'
  home_dir = "./"
  if not os.path.exists(home_dir + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                            extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    os.remove(annotation_zip)
  else:
    annotation_file = './annotations/captions_train2014.json'

  print(annotation_file)

  image_folder = '/train2014/'
  if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                        extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    print(PATH)
    os.remove(image_zip)
  else:
    PATH = home_dir + image_folder
  print(PATH)

  return annotation_file, PATH