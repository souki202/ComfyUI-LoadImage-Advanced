from copy import deepcopy
import time
import numpy as np
import torch
import folder_paths
import os
from PIL import Image, ImageEnhance

from nodes import MAX_RESOLUTION, ImageScale, ImageScaleBy, LatentUpscaleBy, LoadImage, VAEEncode
from .utils import fixing_resolution, rotate_hue_vector

class LoadImageUpscaleBy:
    latent_upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    image_upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "vae": ("VAE", ),
                "image": (sorted(files), {"image_upload": True}),

                "image_upscale_method": (cls.image_upscale_methods,),
                "image_scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),

                "latent_upscale_method": (cls.latent_upscale_methods,),
                "latent_scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),

                "resolution_factor": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            }
        }
        
    CATEGORY = "image"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image, vae, image_upscale_method, image_scale_by, latent_upscale_method, latent_scale_by, resolution_factor):
        (output_image, output_mask) = LoadImage().load_image(image)
        (upscaled_image,) = ImageScaleBy().upscale(output_image, image_upscale_method, image_scale_by) if image_scale_by != 1.0 else (output_image, )

        if resolution_factor > 1:
            upscaled_image = fixing_resolution(upscaled_image, resolution_factor, image_upscale_method)

        (latent,) = VAEEncode().encode(vae, upscaled_image)
        (upscaled_latent,) = LatentUpscaleBy().upscale(latent, latent_upscale_method, latent_scale_by) if latent_scale_by != 1.0 else (latent, )
        return (upscaled_latent, upscaled_image, output_mask)
    
class LoadImageUpscale:
    latent_upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    image_upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]
    
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "vae": ("VAE", ),
                "image": (sorted(files), {"image_upload": True}),

                "image_upscale_method": (cls.image_upscale_methods,),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "crop": (cls.crop_methods,),
                "resolution_factor": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            }
        }
        
    CATEGORY = "image"
    RETURN_TYPES = ("LATENT", "IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image, vae, image_upscale_method, width, height, crop, resolution_factor):
        (output_image, output_mask) = LoadImage().load_image(image)
        (upscaled_image,) = ImageScale().upscale(output_image, image_upscale_method, width, height, crop)
        
        if resolution_factor > 1:
            upscaled_image = fixing_resolution(upscaled_image, 16, image_upscale_method)

        (latent,) = VAEEncode().encode(vae, upscaled_image)
        return (latent, upscaled_image, output_mask)

class ColorAdjustment:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "hue_degree": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.05}),
            }
        }
        
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "color_adjustment"
    def color_adjustment(self, image, hue_degree, contrast, saturation, brightness):
        hue_degree = (hue_degree + 360) % 360
        new_images = torch.zeros_like(image)
        for i in range(len(image)):
            npimg = image[i].numpy()
            rotated_hue = rotate_hue_vector(npimg, hue_degree)

            simple_image = Image.fromarray((np.array(rotated_hue) * 255).astype(np.uint8))
            simple_image = ImageEnhance.Brightness(simple_image).enhance(brightness)
            simple_image = ImageEnhance.Contrast(simple_image).enhance(contrast)
            simple_image = ImageEnhance.Color(simple_image).enhance(saturation)
            new_images[i] = torch.from_numpy(np.array(simple_image).astype(np.float32) / 255).unsqueeze(0)

        return (new_images, )
