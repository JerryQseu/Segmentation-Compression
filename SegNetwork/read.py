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

    out = np.zeros((4,shape[0],shape[1]), seg.dtype)

    out[3] = outs[22]
    out[2] = outs[21] + outs[20]
    out[1] = outs[15] + outs[14] + outs[13] + outs[12] + outs[11] + outs[10]

    for i in range(10):
        out[0] = out[0] + outs[i]
    for i in range(16,20):
        out[0] = out[0] + outs[i]
    for i in range(23,35):
        out[0] = out[0] + outs[i]


    #
    # vals = np.unique(seg)
    # res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    # for c in range(len(vals)):
    #     res[c][seg == c] = 1
    return out
def convert_to_one_hot1(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res

def resize_label(image):
    edsize = image.shape

    # tem = convert_to_one_hot1(image)
    # vals = np.unique(image)

    result = np.zeros((edsize[0],edsize[1]//2,edsize[2]//2), image.dtype)
    for i in range(len(image)):
        result[i,:,:] =resize(image[i].astype(float), ( edsize[1] // 2, edsize[2] // 2), order=1, mode='edge')[None]
    # image = vals[np.vstack(result).argmax(0)]

    return result


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
        ins = img[index].rfind('/')
        # print(img[index])
        # print(img[index][ins:-15] )
        labelname = Path_lab + img[index][ins:-15] + 'gtFine_labelIds.png'
        label = io.imread(labelname)
        label_one = convert_to_one_hot(label)

        edsize = image.shape

        image = resize(image, (edsize[0], edsize[1]//2, edsize[2] // 2), order=3,
                       mode='edge').astype(np.float32)

        imagenorm = normor(image)
        label_one = resize_label(label_one)

        return imagenorm,label_one





train_data = Data(img)
