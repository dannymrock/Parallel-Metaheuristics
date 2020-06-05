import time
import multiprocessing as mp
import sys
import numpy as np
import queue

# Dummy function simulates the execution of a metaheuristic method
def dummy_func (seed, output):
    print(seed)
    np.random.seed(seed)
    for i in range(1,1000000):                  # un gran ciclo
        x = 234.2 / i * np.random.randint(100)  # una división en punto flotante     
    output.put(x) 



if __name__ == '__main__':
    # Get number of cores    
    n = mp.cpu_count()
    print("Number of processors: ", n)

    # Set random seed
    print("seed: ")
    seed = input("Type your seed (none for using current time as seed): ")
    if seed == '':
        seed = int(time.time())
    else:
        seed = int(seed)
    print(seed)
    np.random.seed(seed)


    ### Sequential Execution: we want to execute dummy_func "n" times
    print("Starting sequential execution")
    # Create a simple queue
    fifo_queue = queue.Queue()
    start = time.time()         # taking initial time (from this line)
    for i in range(n):
        dummy_func(np.random.randint(100), fifo_queue)
    end = time.time()           # taking end time
    # Printing results 
    results = [fifo_queue.get() for i in range(n)]
    print(results)

    print('Sequential execution time:', (end - start))


    ### Parallel Execution: we want to execute dummy_func "n" times... in Parallel!
    print("Starting parallel execution")
    
    # Create concurrent queue
    output = mp.Queue()
    # Create parallel activities (objects)
    processes = [mp.Process(target=dummy_func,
                            args=(np.random.randint(100),output))
                 for x in range(n)]

    start = time.time()         # taking initial time (from this line)
    # starting n parallel activities
    for p in processes:
        p.start()

    # Waiting for the termination of the n parallel activities
    for p in processes:
        p.join()
    end = time.time()           # taking end time
    
    results = [output.get() for p in processes]
    print(results)

    print('Parallel execution time:', (end - start))
