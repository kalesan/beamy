import numpy as np

done_cases = np.load('done.npy')

for c in done_cases:
    print("Done: %s" % c)
