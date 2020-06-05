import random
import time
import numpy as np

print("seed: ")
seed = input("Type your seed (none for using current time as seed): ")

if seed == '':
    seed = int(time.time())
else:
	seed = int(seed)

print(seed)

np.random.seed(seed)
print("Seed:", seed)

for x in range(10):
    print (np.random.randint(100))