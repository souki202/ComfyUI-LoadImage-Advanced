import folder_paths
import os

from nodes import MAX_RESOLUTION, ImageScale, ImageScaleBy, LatentUpscaleBy, LoadImage, VAEEncode


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
            }
        }
        
    CATEGORY = "image"
    RETURN_TYPES = ("LATENT", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image, vae, image_upscale_method, image_scale_by, latent_upscale_method, latent_scale_by):
        (output_image, output_mask) = LoadImage().load_image(image)
        (upscaled_image,) = ImageScaleBy().upscale(output_image, image_upscale_method, image_scale_by) if image_scale_by != 1.0 else (output_image, )
        (latent,) = VAEEncode().encode(vae, upscaled_image)
        (upscaled_latent,) = LatentUpscaleBy().upscale(latent, latent_upscale_method, latent_scale_by) if latent_scale_by != 1.0 else (latent, )
        return (upscaled_latent, output_mask)
    
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
            }
        }
        
    CATEGORY = "image"
    RETURN_TYPES = ("LATENT", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image, vae, image_upscale_method, width, height, crop):
        (output_image, output_mask) = LoadImage().load_image(image)
        (upscaled_image,) = ImageScale().upscale(output_image, image_upscale_method, width, height, crop)

        (latent,) = VAEEncode().encode(vae, upscaled_image)
        return (latent, output_mask)
