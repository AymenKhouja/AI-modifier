from ui.image_processing import load_image
from ui.mask_creator import MaskCreator


class Infer: 
    def __init__(self, url = None, image_path = None, segmentation_model = None, not_included = None, pipe = None):
        self.mask = None
        self.not_included = not_included
        image_result = load_image(url,image_path)
        self.init_image = image_result[0]
        self.image_width = image_result[2]
        self.image_height = image_result[1]
        self.mask_creator = MaskCreator(image = self.init_image, segmentation_model = segmentation_model, not_included = self.not_included)
        self.pipe = pipe
    def prepare_mask(self, dilation = 10): 
        labels = self.mask_creator.get_relevant_labels()
        mask = self.mask_creator.create_mask(labels)
        mask = self.mask_creator.dilate_mask(dilation)
        self.mask = mask
        
    def visualize_mask(self): 
        self.mask_creator.visualize_mask()
    
    def get_labels(self): 
        return self.mask_creator.get_detected_labels()
    def __call__(self,prompt, negative_prompt, image = None, mask_image = None, num_inference_steps=30, height = None, width = None,pipe = None, **kwargs): 
        if self.mask is None: 
            self.prepare_mask()
        init_image = image if image is not None else self.init_image
        mask = mask_image if mask_image is not None else self.mask
        image_height = height if height is not None else self.image_height
        image_width = width if width is not None else self.image_width
        image = self.pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=num_inference_steps, image = init_image, mask_image = mask, height = image_height, width = image_width, **kwargs).images[0]
        return image