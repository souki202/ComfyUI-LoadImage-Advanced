
from copy import deepcopy
import math
from typing import List
from nodes import ImageScale

import numpy as np

def rotate_hue_vector(rgb_image: List[List[List[float]]], degree: int) -> List[List[List[float]]]:
    if degree == 0:
        return deepcopy(rgb_image)
    
    radians = np.radians(degree)
    sin = np.sin(radians)
    cos = np.cos(radians)

    # 1/sqrt(3) * (1,1,1)を軸として作成
    n = (1 / math.sqrt(3))
    n2 = n * n
    cos2 = (1 - cos) * n2
    rot = np.array([[cos2 + cos, cos2 - n * sin, cos2 + n * sin],
                    [cos2 + n * sin, cos2 + cos, cos2 - n * sin],
                    [cos2 - n * sin, cos2 + n * sin, cos2 + cos]])

    return [[rot @ rgb for rgb in row] for row in rgb_image]

def fixing_resolution(image, n, upscale_method):
    if n <= 1:
        return image
    width = image.size(2)
    height = image.size(1)
    width = width + (n - width % n) if width % n != 0 else width
    height = height + (n - height % n) if height % n != 0 else height
    return (ImageScale().upscale(image, upscale_method, width, height, "disabled"))[0]
