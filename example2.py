import random
import time
import multiprocessing as mp
import sys

ended_tasks=0

def local_search (seed, output):
    rand = random.Random(seed)
    for i in xrange(1,1000000000):
        x = 234.23423/i     
    output.put(x)
#========================================

# Get number of cores    
cores = mp.cpu_count()
print("Number of processors: ", cores)

# Set random seed
seed = raw_input("Type a seed for your run (default: current system time): ")
if seed == '':
    seed = int(time.time())   
random.seed(seed)
print("Seed:", seed)

#Initialize Pool
output = mp.Queue()

# Create parallel activities
processes = [mp.Process(target=local_search, args=(random.randrange(sys.maxint),output)) for x in range(cores)]

for p in processes:
    p.start()

for p in processes:
    p.join()

results = [output.get() for p in processes]
print(results)

