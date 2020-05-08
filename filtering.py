import os
from os import listdir
import cv2
import numpy as np 
from natsort import natsorted, ns
import glob
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from scipy import ndimage, misc
from skimage import exposure
from scipy.ndimage import gaussian_filter
#from subject_filter_list import *


def Filtering(img, **subject_dict):
	croppedimg = img#[20:276,111:367]#frame[20:400,50:430]
	imggray = cv2.cvtColor(croppedimg, cv2.COLOR_BGR2GRAY)
	img_gaussian = gaussian_filter(imggray, sigma = subject_dict['sigma_'])
	#img_gaussian = gaussian_filter(croppedimg, sigma = subject_dict['sigma_'])
	rescale = exposure.rescale_intensity(img_gaussian, in_range=subject_dict['in_range_'])
	img_histeq = exposure.equalize_hist(rescale)
	img_median = ndimage.median_filter(img_histeq, size=subject_dict['size_'])
	saveimage = cv2.normalize(	src=img_median,
								dst=None,
								alpha=0, 
								beta=255,
								norm_type=cv2.NORM_MINMAX,
								dtype=cv2.CV_8U)
	#ret, thresh = cv2.threshold(saveimage, 55, 255, cv2.THRESH_BINARY_INV)
	return saveimage

filter_60 = dict(sigma_=2,
				in_range_=(40,100),
				size_=20)
filter_80 = dict(sigma_=2,
				in_range_=(50,100),
				size_=20)
filter_98 = dict(	sigma_=2,
				in_range_=(60,100),
				size_=15)