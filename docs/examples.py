import numpy as np

import kastore


def simple_example():
    data = {"one": np.arange(5, dtype=np.int8), "two": np.arange(5, dtype=np.uint64)}
    kastore.dump(data, "tmp.kas")

    d2 = kastore.load("tmp.kas")
    print(list(d2.items()))


simple_example()
