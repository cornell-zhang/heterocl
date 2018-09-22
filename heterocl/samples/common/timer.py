from timeit import default_timer as timer
import numpy as np

def timeit(func, args=(), num_round=100, iter_per_round=10):
  x = [None] * num_round
  y = [iter_per_round * _ for _ in range(num_round)]
  for i in range(num_round):
    start = timer()
    for j in range(iter_per_round * i):
      func(*args)

    end = timer()
    x[i] = (end - start) * 1e9  # nanoseconds

  return np.polyfit(x, y, 1)
