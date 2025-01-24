from abc import ABC, abstractmethod
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image
import requests
from typing import Tuple, Optional, Union
from io import BytesIO

class ImageUtils:
    """Utility class for image processing tasks."""

    @staticmethod
    def convert_to_numpy(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Converts a PIL Image or any supported type to a NumPy array."""
        if isinstance(image, np.ndarray):
            return image
        try:
            return np.array(image)
        except Exception as e:
            raise ValueError(
                f"Cannot convert object of type {type(image)} to a numpy array: {e}"
            )

    @staticmethod
    def calculate_dimensions(
        original_width: int,
        original_height: int,
        width: Optional[int],
        height: Optional[int],
    ) -> Tuple[int, int]:
        """Calculates new dimensions for resizing, ensuring divisibility by 8."""
        if width is None and height is None:
            return original_width, original_height

        aspect_ratio = original_width / original_height

        if width is not None:
            new_width = width
            new_height = int(new_width / aspect_ratio)
        elif height is not None:
            new_height = height
            new_width = int(new_height * aspect_ratio)
        else:
            new_height, new_width = height, width

        # Ensure dimensions are divisible by 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        return new_width, new_height


# Abstract class for image loading
class ImageLoader(ABC):
    """Abstract base class for loading images."""

    @abstractmethod
    def load_image(self) -> Image.Image:
        """Loads an image and returns it."""
        raise NotImplementedError


# File-based Image Loader
class FileImageLoader(ImageLoader):
    """Loads images from the file system."""

    def load_image(self, image_path: str = None) -> Image.Image:
        """Loads an image from the local file system."""
        if not image_path:
            image_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")]
            )
        if image_path:
            return Image.open(image_path)
        raise FileNotFoundError("No file selected or invalid path.")


# URL-based Image Loader
class URLImageLoader(ImageLoader):
    """Loads images from a URL."""

    def load_image(self, url: str) -> Image.Image:
        """Loads an image from the given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")


# Separate class for image resizing
class ImageResizer:
    """Handles resizing of images."""

    @staticmethod
    def image_resize(
        image: Union[np.ndarray, Image.Image],
        width: Optional[int] = None,
        height: Optional[int] = None,
        inter: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """
        Resizes an image, preserving aspect ratio, and ensuring that the width or
        height (if specified) are divisible by 8.

        Args:
            image : Input image in either PIL or NumPy array format.
            width : Target width (must be divisible by 8). If None, infer from height.
            height : Target height (must be divisible by 8). If None, infer from width. 
            inter : Interpolation method for resizing. Default is cv2.INTER_AREA.

        Returns:
            Resized image as a NumPy array.
        """

        image = ImageUtils.convert_to_numpy(image)

        original_height, original_width = image.shape[:2]

        if width is None and height is None:
            return Image.fromarray(image), original_height, original_width

        if width == original_width and height == original_height:
            return image

        new_width, new_height = ImageUtils.calculate_dimensions(
            original_width, original_height, width, height
        )

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=inter)
        resized_image = Image.fromarray(resized_image)
        return resized_image, new_height, new_width


# 5. The Facade: ImageHandler Class
class ImageHandler:
    """High-level interface for handling images, including loading, 
    resizing, and converting."""

    def __init__(self, loader: ImageLoader, resizer: ImageResizer = None):
        self.loader = loader
        self.resizer = resizer if resizer else ImageResizer()

    def load_resize_image(
        self, source: str = None, width: int = None, height: int = None
    ) -> np.ndarray:
        """Loads, resizes, and converts an image to a numpy array."""
        image = self.loader.load_image(source)

        if width or height:
            return self.resizer.image_resize(image, width, height)

        return image, image.size[1], image.size[0]
