
from copy import deepcopy
import math
from typing import List

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
