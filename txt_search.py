import glob
import pickle
import shutil
import os

data_path = 'data/top_5/train' # Put the directory of the data

"""img_id_dict = {}
for img_path in glob.iglob('%s/*/*/*' % data_path):
    if not img_path.endswith('RGB.png'):
        continue
    
    img_cls = img_path.split('\\')[-3]
    # E.g. '<img_idx>_<img_id>_<img_type>.png'
    img_name = img_path.split('\\')[-1]
    img_var = img_name.split('_')[0]
    img_id = img_name.split('_')[1]
    img_id_with_var = img_var + '_' + img_id
    img_id_dict[img_id_with_var] = img_cls
pickle.dump(img_id_dict, open('img_id_collection.obj', 'wb'))
print(len(list(img_id_dict.keys())))
input()"""

# Create grasp data folder
os.mkdir('grasps')
cls_list = []
with open('cls.txt', 'r') as f:
    file = f.readlines()
    for cls in file:
        # remove '\n' from string
        cls = cls[:-1]
        cls_list.append(cls)
for cls in cls_list:
    os.mkdir('grasps/%s' % cls)

img_id_dict = pickle.load(open('img_id_collection.obj', 'rb'))
img_id_set = set(img_id_dict.keys())
print('Number of searches: ' + len(img_id_set))

# The number of '*/' depends on the structure of the dataset
for img_path in glob.iglob('%s/*/*/*' % data_path):
    # If condition to avoid counting the irrelevant files.
    if not img_path.endswith('grasps.txt'):
        continue
    
    # E.g. '<img_idx>_<img_id>_<img_type>.png'
    img_name = img_path.split('\\')[-1]
    img_var = img_name.split('_')[0]
    img_id = img_name.split('_')[1]
    img_id_with_var = img_var + '_' + img_id
    img_grasp_name = img_id_with_var + '_grasps.txt'

    if img_id_with_var in img_id_set:
        img_cls = img_id_dict[img_id_with_var]
        if img_id not in os.listdir('grasps/%s' % img_cls):
            os.mkdir('grasps/%s/%s' % (img_cls, img_id))
        shutil.copyfile(img_path, 'grasps/%s/%s/%s' % (img_cls, img_id, img_grasp_name))
