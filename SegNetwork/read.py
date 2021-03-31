import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
import torch.utils.data.dataset as Dataset

def normor(image): #[n,d,w,h]
    image -=image.mean()
    image /=image.std()
    return image

def convert_to_one_hot(seg):
    shape = seg.shape
    outs = np.zeros((35,shape[0],shape[1]), seg.dtype)

    for i in range(35):

        outs[i][seg == i] = 1

    outs[34][seg==-1] = 1


    #
    # vals = np.unique(seg)
    # res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    # for c in range(len(vals)):
    #     res[c][seg == c] = 1
    return outs

Path_img = './data/image/'
Path_lab = './data/label/'

img = []

for root,dirs,files in os.walk(Path_img):
    for file in files:
        fourd_path = os.path.join(root, file)

        img.append(fourd_path)

#
# image = io.imread(img[0]).astype(float)
# image = image.transpose(2,0,1)
# imagenorm = normor(image)
# labelname = Path_lab + img[0][-36:-15] + 'gtCoarse_labelIds.png'
# label = io.imread(labelname)
#
# label_one = convert_to_one_hot(label)

class Data(Dataset.Dataset):
    def __init__(self,img):
        self.img = img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):

        image = io.imread(self.img[index]).astype(float)
        image = image.transpose(2, 0, 1)
        labelname = Path_lab + img[0][-36:-15] + 'gtCoarse_labelIds.png'
        label = io.imread(labelname)
        label_one = convert_to_one_hot(label)

        imagenorm = normor(image)

        return imagenorm,label_one





train_data = Data(img)
