import cv2
import numpy as np
from PIL import Image
import requests

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    if type(image) != np.ndarray:
        image = np.array(image)
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        assert height % 8 == 0, "Height must be divisible by 8"
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        width = int(w*r)
        if width % 8 != 0: 
            width = width - width % 8
        dim = (width, height)
        
    # otherwise, the height is None
    else:
        assert width % 8 == 0, "Width must be divisible by 8"
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
       
        height = int(h * r)
        if height % 8 != 0:
            height = height - height % 8
        dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = inter)
    
    return resized


def load_image(url, image_path): 
    if url is not None:
        image = Image.open(requests.get(url, stream=True).raw)
    elif image_path is not None:
        image = Image.open(image_path)
    w, h = image.size
    print("Image shape before resizing:", image.size)
    if w >= 800:
        image = image_resize(image, width = 800)
        h = image.shape[0]
        w = image.shape[1]
        image = Image.fromarray(image)
    elif w < 800 and w % 8 != 0:
        image = image_resize(image, width = w - w % 8)
        image = Image.fromarray(image)
    elif w < 800 and h % 8 != 0:
        image = image_resize(image, height = h - h % 8)
        image = Image.fromarray(image)
    print("Image shape after resizing:", image.size)
    w, h = image.size
    return image, h, w
