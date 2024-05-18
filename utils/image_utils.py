from tkinter import filedialog
from PIL import Image, ImageTk
import requests
from io import BytesIO


def resize_image(image, resize_dim):
    width = resize_dim[0]
    height = resize_dim[1] 
    return image.thumbnail(width, height)

def load_image(resize_dim :tuple = None):
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
    if filename:
        image = Image.open(filename)
        if resize_dim is not None:
            image = resize_image(image, resize_dim)
        return 


def load_from_url(url_entry, status_label, resize_dim :  tuple = None):
    url = url_entry.get()
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        status_label.config(text="Failed to load image from URL")