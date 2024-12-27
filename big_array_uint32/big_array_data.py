import numpy as np

rng = np.random.RandomState(42)
MAX = np.iinfo(np.uint32).max
a = rng.randint(0, MAX, int(1e9), dtype=np.uint32)
print(a[0:10])

with open("big_array.bin", "wb") as fout:
    fout.write(a.tobytes())
