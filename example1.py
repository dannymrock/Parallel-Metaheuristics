import random
import time

print("seed: ")
seed = raw_input("Type your seed (none for using current time as seed): ")
#print("input ", seed)
if seed == '':
    seed = int(time.time())

random.seed(seed)
print("Seed:", seed)

for x in range(10):
    print random.randint(1,101)

