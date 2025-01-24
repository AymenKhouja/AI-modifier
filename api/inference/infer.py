from abc import ABC, abstractmethod
from ..utils.image_processing import (
    ImageHandler,
    FileImageLoader,
    URLImageLoader,
    ImageResizer,
)
from api.inference.mask_creator import MaskCreator, HuggingfaceSegmentationManager
from transformers import ImageSegmentationPipeline, pipeline
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector


openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


class BasicInpaintInfer(ABC):
    def __init__(
        self,
        segmentation_model=None,
        not_included=None,
        relevant_labels=None,
        pipe=None,
    ):
        default_segmentation_model = segmentation_model
        segmentation_manager = self._set_segmentation_manager(
            default_segmentation_model
        )
        self.mask_creator = MaskCreator(
            segmentation_manager=segmentation_manager,
            not_included=not_included,
            relevant_labels=relevant_labels,
        )

        self.pipe = pipe
        # Initialize attributes to track image handling
        self.image_handler = None
        self.current_source_type = None
        self.mask_data = None

    def _set_segmentation_manager(self, segmentation_model):
        """Automatically sets the segmentation manager based on the provided model."""
        if isinstance(segmentation_model, str):
            # Assume it's a Huggingface model string path
            return HuggingfaceSegmentationManager(
                pipeline("image-segmentation", segmentation_model)
            )
        elif isinstance(segmentation_model, ImageSegmentationPipeline):
            return HuggingfaceSegmentationManager(segmentation_model)
        else:
            return HuggingfaceSegmentationManager(segmentation_model)

    def _set_image_handler(self, source_type):
        """Sets the appropriate ImageHandler based on the source type."""
        if source_type == "url":
            self.image_handler = ImageHandler(
                loader=URLImageLoader(), resizer=ImageResizer()
            )
        else:  # Default to file handling
            self.image_handler = ImageHandler(
                loader=FileImageLoader(), resizer=ImageResizer()
            )

    @abstractmethod
    def load_image(self, source, width=None, height=None, source_type="file"):
        raise NotImplementedError

    def prepare_mask(self, dilation=10, invert_mask=False, visualize=True):
        """Creates a mask for the current image, with an option to invert it."""
        # Generate the initial mask
        self.mask_data = self.mask_creator.create_mask(
            self.current_image, dilation=dilation, visualize=visualize
        )

        # Invert the mask if specified
        if invert_mask:
            self.mask_data.mask_image = Image.fromarray(
                255 - np.array(self.mask_data.mask_image)
            )

    @abstractmethod
    def __call__(
        self,
        prompt,
        negative_prompt,
        image=None,
        mask_image=None,
        num_inference_steps=30,
        height=None,
        width=None,
        pipe=None,
        **kwargs
    ):
        raise NotImplementedError


class SimpleInpaintInfer(BasicInpaintInfer):
    def __init__(
        self,
        segmentation_model=None,
        not_included=None,
        relevant_labels=None,
        pipe=None,
    ):
        super().__init__(
            segmentation_model=segmentation_model,
            not_included=not_included,
            relevant_labels=relevant_labels,
            pipe=pipe,
        )

    def load_image(self, source, width=None, height=None, source_type="file"):
        """Loads and resizes an image based on the provided source type."""
        # Change image handler only if source_type has changed
        if self.current_source_type != source_type:
            self._set_image_handler(source_type)
            self.current_source_type = source_type

        # Load and resize the base image
        (
            self.current_image,
            self.image_height,
            self.image_width,
        ) = self.image_handler.load_resize_image(source, width, height)

    def __call__(
        self,
        prompt,
        negative_prompt,
        image=None,
        mask_image=None,
        num_inference_steps=30,
        height=None,
        width=None,
        pipe=None,
        **kwargs
    ):
        if self.mask_data is None and mask_image is None:
            self.prepare_mask()

        pipe = pipe if pipe is not None else self.pipe
        init_image = image if image is not None else self.current_image
        mask = mask_image if mask_image is not None else self.mask_data.mask_image
        image_height = height if height is not None else self.image_height
        image_width = width if width is not None else self.image_width

        # Generate the inpainted image using the pipe
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            image=init_image,
            mask_image=mask,
            height=image_height,
            width=image_width,
            **kwargs
        ).images[0]

        return image


class OutpaintInpaintInfer(SimpleInpaintInfer):
    def __init__(
        self,
        segmentation_model=None,
        not_included=None,
        relevant_labels=None,
        pipe=None,
    ):
        super().__init__(
            segmentation_model=segmentation_model,
            not_included=not_included,
            relevant_labels=relevant_labels,
            pipe=pipe,
        )

    def load_image(
        self,
        source,
        width=None,
        height=None,
        source_type="file",
        outpaint_dimensions=None,
        position="left",
    ):
        """Loads and resizes an image based on the provided source type, 
        with optional outpainting."""
        # Change image handler only if source_type has changed
        if self.current_source_type != source_type:
            self._set_image_handler(source_type)
            self.current_source_type = source_type

        # Load and resize the base image
        (
            self.current_image,
            self.image_height,
            self.image_width,
        ) = self.image_handler.load_resize_image(source, width, height)

        # Apply outpainting if specified
        if outpaint_dimensions:
            self.current_image = self._outpaint_image(
                self.current_image, outpaint_dimensions, position
            )
            self.image_width, self.image_height = self.current_image.size

    def _outpaint_image(self, image, outpaint_dimensions, position):
        """Expands the image canvas and aligns the original image, 
        filling surrounding areas with white."""
        target_width, target_height = outpaint_dimensions

        # Ensure target dimensions are at least as equal as the original
        target_width = image.width if target_width < image.width else target_width
        target_height = image.height if target_height < image.height else target_height

        # Create a new image with the desired dimensions, filled with white
        new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))

        # Determine the offsets
        offset_x = (target_width - image.width) // 2
        if position == "left":
            offset_x = 0
        elif position == "right":
            offset_x = target_width

        offset_y = (
            0 if target_height > image.height else (target_height - image.height) // 2
        )  # Align to top if only height grows

        # Paste the original image onto the new canvas
        new_image.paste(image, (offset_x, offset_y))

        # Update offsets for mask alignment
        self.mask_offset = (offset_x, offset_y)

        return new_image
