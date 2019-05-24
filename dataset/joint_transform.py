"""This function is used for data augument for npy dataset"""


import random
from utils.show_results import *



class JointCompose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        for t in self.transforms:
            img,mask=t(img,mask)
        return img,mask

# the selected augument
class JointRandomFlip(object):
    """No flip along the thrid axis: 0.2 0.2 1"""
    def __call__(self,img,mask):
        p=0.5
        # axis=random.randint(0,1)
        # axis = random.randint(0, 2)

        axis = random.randint(1, 2)
        if random.random()<p:
            img=np.flip(img,axis).copy()
            mask=np.flip(mask,axis).copy()
        return img,mask


class JointRandomRotation(object):
    def __call__(self, img, mask):
        ang=random.randint(1,3)
        # axis=random.sample([0,1,2],2)

        axis=(1,2)

        p=0.5
        if random.random()<p:
            img=np.rot90(img,ang,axis).copy()
            mask=np.rot90(mask,ang,axis).copy()

        return img,mask


class JointRandomGaussianNoise(object):
    """define the noise amplitude and weight : img float and label long"""
    def __init__(self,amplitude=10):
        self.amplitude=amplitude

    def __call__(self, img, mask):
        # eliminate the std of the image
        nlevel=random.random()*self.amplitude/350

        p=0.5
        if random.random()<p:
            # print('transform nlevel:{}'.format(nlevel))
            noise=nlevel*np.random.normal(0,1,img.shape)
            img=img+noise

        return img,mask


class JointRandomSubTractGaussianNoise(object):
    """define the noise amplitude and weight : img float and label long"""
    def __init__(self,amplitude=10):
        self.amplitude=amplitude

    def __call__(self, img,mask):
        nlevel=random.random()*self.amplitude/350

        p=0.5
        if random.random()<p:
            # print(nlevel)
            noise=nlevel*np.random.normal(0,1,img.shape)
            img=img-noise

        return img,mask



class JointRandomBrightness(object):
    def __init__(self,limit):
        self.limit=limit

    def __call__(self, img,mask):
        p=0.5
        if random.random()<p:
            alpha=1.0+np.random.uniform(self.limit[0],self.limit[1])
            # print(alpha)
            img=alpha*img
        return img,mask


class JointRandomIntensityChange(object):
    def __init__(self,limit):
        self.limit=limit

    def __call__(self, img,mask):
        p=0.5
        if random.random()<p:
            alpha=np.random.uniform(self.limit[0],self.limit[1])
            # print(alpha)
            img=img+alpha
        return img,mask






"""For single image"""

class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img):
        for t in self.transforms:
            img=t(img)
        return img

# the selected augument
class RandomFlip(object):
    """No flip along the thrid axis: 0.2 0.2 1"""
    def __call__(self,img):
        p=0.5
        axis=random.randint(0,1)
        if random.random()<p:
            img=np.flip(img,axis).copy()
        return img


class RandomRotation(object):
    def __call__(self, img):
        ang=random.randint(1,3)
        axis=random.sample([0,1,2],2)

        p=0.5
        if random.random()<p:
            img=np.rot90(img,ang,axis).copy()

        return img


class RandomGaussianNoise(object):
    """define the noise amplitude and weight : img float and label long"""
    def __init__(self,amplitude=10):
        self.amplitude=amplitude

    def __call__(self, img):
        nlevel=random.random()*self.amplitude/350

        p=0.5
        if random.random()<p:
            # print(nlevel)
            noise=nlevel*np.random.normal(0,1,img.shape)
            img=img+noise

        return img


class RandomSubTractGaussianNoise(object):
    """define the noise amplitude and weight : img float and label long"""
    def __init__(self,amplitude=10):
        self.amplitude=amplitude

    def __call__(self, img):
        nlevel=random.random()*self.amplitude/350

        p=0.5
        if random.random()<p:
            # print(nlevel)
            noise=nlevel*np.random.normal(0,1,img.shape)
            img=img-noise

        return img



class RandomBrightness(object):
    def __init__(self,limit):
        self.limit=limit

    def __call__(self, img):
        p=0.5
        if random.random()<p:
            alpha=1.0+np.random.uniform(self.limit[0],self.limit[1])
            # print(alpha)
            img=alpha*img
        return img


class RandomIntensityChange(object):
    def __init__(self,limit):
        self.limit=limit

    def __call__(self, img):
        p=0.5
        if random.random()<p:
            alpha=np.random.uniform(self.limit[0],self.limit[1])
            # print(alpha)
            img=img+alpha
        return img


