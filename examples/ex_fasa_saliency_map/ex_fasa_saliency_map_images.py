#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io
# https://mpatacchiola.github.io/blog/
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# In this example the FASA algorithm is used in order to process some images.
# The original image and the saliency version are showed for comparison.

import numpy as np
import cv2
from timeit import default_timer as timer
from deepgaze.saliency_map import FasaSaliencyMapping


def main(images_path):
    '''image_1 = cv2.imread(path1)
    image_2 = cv2.imread(path2)
    image_3 = cv2.imread(path3)
    image_4 = cv2.imread(path4)'''

    images = []

    for image in images_path:
        images.append(cv2.imread(image))

    i = 0
    image_salient = []

    for image in images:

        my_map = FasaSaliencyMapping(image.shape[0], image.shape[1])  # init the saliency object
        start = timer()
        mask = my_map.returnMask(image, tot_bins=8, format='BGR2LAB')  # get the mask from the original image
        image_salient.append(cv2.GaussianBlur(mask, (3, 3), 1)) # applying gaussin blur to make it pretty
        end = timer()
        i = i+1
        print("--- %s Image %s tot seconds ---" % (end - start, i))

    original_images_stack = np.hstack((images))
    saliency_images_stack = np.hstack((image_salient))
    saliency_images_stack = np.dstack((saliency_images_stack, saliency_images_stack, saliency_images_stack))
    cv2.imwrite("Original-Saliency.jpg", np.concatenate((original_images_stack, saliency_images_stack)))


