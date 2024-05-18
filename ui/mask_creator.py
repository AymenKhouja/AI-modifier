from transformers import pipeline
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import cv2


class MaskCreator:
    def __init__(self, image, segmentation_model=None, not_included=None):
        self.segmenter = segmentation_model or pipeline("image-segmentation", "mattmdjaga/segformer_b2_clothes")
        self.init_image = image
        self.results = None
        self.mask = None
        self.not_included = not_included or []
        self.labels = None

    def segment_image(self):
        return self.segmenter(self.init_image)

    def get_detected_labels(self):
        if self.results is None:
            self.results = self.segment_image()
        labels = [result["label"].lower() for result in self.results]
        self.labels = labels
        return labels

    def get_relevant_labels(self):
        if self.labels is None:
            self.get_detected_labels()
        return [label for label in self.labels if label not in self.not_included]

    def change_results_format(self, results):
        mask_dict = {}
        labels = []
        for result in results:
            labels.append(result["label"])
            mask_dict[result["label"].lower()] = result["mask"]
        self.labels = labels
        return mask_dict

    def preprocess_mask(self, mask):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        three_channel_image = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        three_channel_image = three_channel_image.astype(np.uint8)
        mask = Image.fromarray(three_channel_image)
        return mask

    def create_mask(self, labels):
        if self.results is None:
            self.results = self.segment_image()
        mask_dict = self.change_results_format(self.results)
        height, width = np.array(self.init_image).shape[:2]
        if isinstance(labels, list):
            mask_image = np.zeros((height, width))
            for label in labels:
                mask_image += mask_dict[label]
        else:
            mask_image = mask_dict[labels]
        self.mask = mask_image
        return self.preprocess_mask(self.mask)

    def dilate_mask(self, edge_width):
        kernel = np.ones((edge_width, edge_width), np.uint8)
        dilated_mask = cv2.dilate(self.mask, kernel, iterations=1)
        self.mask = dilated_mask
        return self.preprocess_mask(self.mask)

    def visualize_mask(self):
        mask = self.mask
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        image = self.init_image
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Display original image
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        # Display mask overlaid on original image
        plt.imshow(mask, alpha=0.5)
        plt.title('Image with Mask Overlay')
        plt.axis('off')
        plt.show()
    def compose_image_mask(self): 
        mask_image = self.preprocess_mask(self.mask).convert("RGBA")
        # Create a new mask image with alpha channel
        alpha_mask = np.array(mask_image)
        alpha_channel = alpha_mask[:, :, 0]  # Assuming mask is a grayscale image
        
        alpha_mask[:, :, 3] = np.where(alpha_channel == 255, 255, 0) # Set alpha values based on the mask
        alpha_image = Image.fromarray(alpha_mask, mode='RGBA')
        composed = Image.alpha_composite(self.init_image.convert("RGBA"), alpha_image)
        
        return composed