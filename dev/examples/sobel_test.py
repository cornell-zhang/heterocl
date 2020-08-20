from sobel_hcl import npF
from sobel_filter import newimg

import numpy as np

np.testing.assert_array_equal(npF, newimg)
