from tkinter import filedialog
from PIL import Image
import requests
from io import BytesIO
import numpy as np


def resize_image(image, resize_dim):
    width = resize_dim[0]
    height = resize_dim[1]
    return image.thumbnail(width, height)


def load_image(resize_dim: tuple = None):
    filename = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")]
    )
    if filename:
        image = Image.open(filename)
        if resize_dim is not None:
            image = resize_image(image, resize_dim)
        return


def load_from_url(url_entry, status_label, resize_dim: tuple = None):
    url = url_entry.get()
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        status_label.config(text="Failed to load image from URL")


def check_which_mask(mask_dict, cursor_x, cursor_y):
    dummy_mask = np.array(list(mask_dict.values())[0])
    image_height, image_width = dummy_mask.shape[0], dummy_mask.shape[1]
    cursor_x = int((cursor_x) * image_width)
    cursor_y = int((cursor_y) * image_height)

    for key, value in mask_dict.items():
        image_array = np.array(value)
        if image_array[cursor_y, cursor_x] == 255:
            return key
