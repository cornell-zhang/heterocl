from canny_hcl import finalImg
from canny_filter import resImg

import numpy as np

np.testing.assert_array_equal(finalImg, resImg)
