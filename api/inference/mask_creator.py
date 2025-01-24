from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from transformers import ImageSegmentationPipeline
from dataclasses import dataclass

@dataclass
class MaskMetadata:
    labels: List[str]
    mask_dict : dict
    width : int
    height : int 
    mask_image : np.ndarray


class ISegmentationManager(ABC):
    """Segmentation Manager Interface to enforce decoupling."""
    
    @abstractmethod
    def segment(self, image):
        raise NotImplementedError

class HuggingfaceSegmentationManager(ISegmentationManager):
    """Handles loading and applying Huggingface segmentation models."""
    
    def __init__(self, models: Union[List[ImageSegmentationPipeline], ImageSegmentationPipeline] = None):
        self.segmenters = self._load_segmenters(models)

    def _load_segmenters(self, models: Union[List[ImageSegmentationPipeline], ImageSegmentationPipeline]):
        """Load the segmentation models."""
        if isinstance(models, ImageSegmentationPipeline):
            return [models]
        elif isinstance(models, list) and all(isinstance(model, ImageSegmentationPipeline) for model in models):
            return models
        raise ValueError("All models must be instances of transformers ImageSegmentationPipeline.")

    def segment(self, image):
        """Segment the image using all loaded models."""
        results = []
        for segmenter in self.segmenters:
            results.extend(segmenter(image))
        return results
class MaskFilter:
    """Filters masks based on relevant or non-relevant labels."""
    
    def __init__(self, results: List[dict], not_included: Optional[List[str]] = None):
        self.results = results
        self.not_included = not_included or []
        self.labels = [result["label"].lower() for result in self.results]

    def get_relevant_labels(self, relevant_labels: Optional[List[str]] = None):
        """Filter out non-relevant labels."""
        if relevant_labels:
            return relevant_labels
        return [label for label in self.labels if label not in self.not_included]

    def generate_mask_dict(self, relevant_labels: Optional[List[str]] = None):
        """Generate a dictionary of label-to-mask."""
        relevant = self.get_relevant_labels(relevant_labels)
        return {result["label"].lower(): result["mask"] for result in self.results if result["label"].lower() in relevant}

class MaskGenerator:
    """Handles mask creation and manipulation."""
    
    def __init__(self, mask_filter: MaskFilter):
        self.mask_filter = mask_filter

    def _compose_mask(self, mask_data: MaskMetadata) -> MaskMetadata:
        mask_data.mask_image = np.zeros((mask_data.height, mask_data.width), dtype=np.uint8)
        print(mask_data.labels)
        for label in mask_data.labels:
            mask = mask_data.mask_dict.get(label, np.zeros_like(mask_data.mask_image))
            mask_data.mask_image = np.maximum(mask_data.mask_image, mask)
        return mask_data

    
    def _create_mask_metadata(self, labels): 
        labels = labels or self.mask_filter.get_relevant_labels()
        mask_dict = self.mask_filter.generate_mask_dict()
        width, height = next(iter(mask_dict.values())).size[:2]
        mask_image = np.zeros((height, width))
        return MaskMetadata(labels,mask_dict,width,height,mask_image)
    
    def create_mask(self, labels: Optional[Union[str, List[str]]] = None):
        """Create a binary mask based on selected labels."""
        mask_data = self._create_mask_metadata(labels)
        return self._compose_mask(mask_data)
    
    def dilate_mask(self, mask: np.ndarray, edge_width: int):
        kernel = np.ones((edge_width, edge_width), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

class ImageVisualizer:
    """Handles visualizing the image and mask."""

    @staticmethod
    def display_images(original_image: Image.Image, mask_image: np.ndarray):
        """Display the original image and mask overlay."""
        plt.imshow(original_image)
        plt.imshow(mask_image, alpha=0.5)
        plt.title("Image with Mask Overlay")
        plt.axis("off")
        plt.show()

    @staticmethod
    def compose_image_with_mask(image: Image.Image, mask: np.ndarray):
        """Compose an image with an RGBA mask."""
        mask_image = Image.fromarray(mask).convert("RGBA")
        alpha_channel = np.array(mask_image)[:, :, 0]

        alpha_mask = np.array(mask_image)
        alpha_mask[:, :, 3] = np.where(alpha_channel == 255, 100, 0)

        alpha_image = Image.fromarray(alpha_mask, mode="RGBA")
        composed = Image.alpha_composite(image.convert("RGBA"), alpha_image)

        return composed


class MaskCreator:
    """Coordinates segmentation, mask creation, and visualization."""
    
    def __init__(self, 
                 segmentation_manager: ISegmentationManager, 
                 not_included: Optional[List[str]] = None,
                 relevant_labels: Optional[List[str]] = None):
        self.segmenter = segmentation_manager
        self.not_included = not_included or []
        self.relevant_labels = relevant_labels or []
        self.current_mask = None

    def create_mask(self, image : Image.Image, visualize: bool = False, dilation: int = 0):
        """Perform segmentation, mask creation, and visualize the result."""
        segmentation_results = self.segmenter.segment(image)
        mask_filter = MaskFilter(segmentation_results, not_included=self.not_included)
        mask_generator = MaskGenerator(mask_filter)

        # Generate mask based on relevant labels
        self.mask_data = mask_generator.create_mask(self.relevant_labels)
        self.current_mask = self.mask_data.mask_image

        # Display images
        if visualize:
            ImageVisualizer.display_images(image, self.current_mask)

        # Dilate the mask using the selected strategy
        if dilation:
            self.mask_data.mask_image = mask_generator.dilate_mask(self.current_mask, edge_width=dilation)
            return self.mask_data
        
        return self.mask_data

        