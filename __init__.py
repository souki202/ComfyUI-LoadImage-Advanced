from .load_image_advanced import ColorAdjustment, LoadImageUpscale, LoadImageUpscaleBy

NODE_CLASS_MAPPINGS = {
    "LoadImageUpscaleBy": LoadImageUpscaleBy,
    "LoadImageUpscale": LoadImageUpscale,
    "ColorAdjustment": ColorAdjustment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageUpscaleBy": "Load Image Upscale By",
    "LoadImageUpscale": "Load Image Upscale",
    "ColorAdjustment": "Color Adjustment",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
