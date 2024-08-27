from .load_image_advanced import LoadImageUpscale, LoadImageUpscaleBy

NODE_CLASS_MAPPINGS = {
    "LoadImageUpscaleBy": LoadImageUpscaleBy,
    "LoadImageUpscale": LoadImageUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageUpscaleBy": "Load Image Upscale By",
    "LoadImageUpscale": "Load Image Upscale",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
