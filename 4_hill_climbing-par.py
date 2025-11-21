"""Parallel hill-climbing search for the Tai20b QAP instance."""
import multiprocessing as mp
import time
from typing import Iterable, List, Tuple

import numpy as np

MAX_ITERS = 100
SIZE = 20
NEIGHBORHOOD_SIZE = SIZE * (SIZE - 1) // 2


# Tai20b 20 122455319 (OPT) (8,16,14,17,4,11,3,19,7,9,1,15,6,13,10,2,5,20,18,12)
# zero-base:                (7,15,13,16,3,10,2,18,6,8,0,14,5,12,9 ,1,4,19,17,11)
flow = [[0,23,29,36,46,53,32,28,64,13,65,36,48,8,47,27,761,812,795,795],
       [23,0,8,33,42,45,10,25,48,13,51,15,32,27,26,37,738,789,772,772],
       [29,8,0,40,49,52,3,33,51,21,55,7,35,31,25,45,732,782,765,765],
       [36,33,40,0,10,17,40,9,34,27,32,46,23,44,35,19,750,803,787,784],
       [46,42,49,10,0,8,48,19,30,37,26,54,24,54,39,26,749,802,787,782],
       [53,45,52,17,8,0,51,25,23,43,19,56,22,60,38,34,743,797,781,776],
       [32,10,3,40,48,51,0,33,49,23,53,6,33,34,22,46,729,780,762,762],
       [28,25,33,9,19,25,33,0,39,18,38,39,25,35,33,17,750,802,786,784],
       [64,48,51,34,30,23,49,39,0,52,7,53,17,71,29,53,719,774,758,753],
       [13,13,21,27,37,43,23,18,52,0,53,27,35,19,35,25,750,801,784,784],
       [65,51,55,32,26,19,53,38,7,53,0,57,20,72,34,51,725,779,764,758],
       [36,15,7,46,54,56,6,39,53,27,57,0,38,37,25,52,725,776,759,759],
       [48,32,35,23,24,22,33,25,17,35,20,38,0,54,17,41,727,780,764,760],
       [8,27,31,44,54,60,34,35,71,19,72,37,54,0,52,35,762,813,795,796],
       [47,26,25,35,39,38,22,33,29,35,34,25,17,52,0,50,718,770,753,751],
       [27,37,45,19,26,34,46,17,53,25,51,52,41,35,50,0,767,819,803,801],
       [761,738,732,750,749,743,729,750,719,750,725,725,727,762,718,767,0,73,86,34],
       [812,789,782,803,802,797,780,802,774,801,779,776,780,813,770,819,73,0,32,53],
       [795,772,765,787,787,781,762,786,758,784,764,759,764,795,753,803,86,32,0,77],
       [795,772,765,784,782,776,762,784,753,784,758,759,760,796,751,801,34,53,77,0]]

distance=[[0, 1341, 283, 17514, 0, 5387, 10, 0, 0, 0, 17307, 98, 122, 1325, 0, 0, 378, 239, 1, 1],
          [0, 0, 336, 0, 1, 0, 0, 3, 0, 1, 2109, 0, 130, 0, 0, 0, 2, 0, 0, 241],
          [5134, 0, 0, 0, 5811, 0, 0, 458, 11, 127, 0, 18012, 28, 0, 7, 0, 97, 4, 23, 0],
          [3896, 0, 0, 0, 2, 41453, 206, 6590, 16724, 0, 0, 375, 0, 0, 2474, 1868, 43487, 0, 0, 0],
          [8, 0, 0, 35, 0, 18, 0, 2447, 0, 23862, 168, 55344, 0, 0, 22946, 0, 1, 9818, 0, 582],
          [0, 3, 1009, 635, 14785, 0, 20, 52, 0, 24683, 743, 0, 0, 1, 0, 0, 2, 0, 1, 1],
          [54212, 1, 0, 7, 3443, 0, 0, 0, 59, 955, 0, 0, 0, 0, 2, 6, 0, 0, 29, 0],
          [1, 0, 38645, 11, 0, 31, 24, 0, 8407, 2101, 0, 1, 0, 0, 0, 2142, 61, 0, 37, 1316],
          [0, 251, 52, 0, 0, 0, 2, 0, 0, 0, 5, 3, 9610, 1608, 3021, 397, 10, 2, 0, 1],
          [0, 0, 0, 7, 0, 0, 4152, 0, 0, 0, 9, 1, 1, 0, 7, 0, 2, 344, 3723, 2242],
          [1945, 0, 3007, 53507, 0, 1, 63, 0, 0, 1150, 0, 0, 0, 0, 268, 276, 0, 0, 0, 0],
          [1038, 2, 670, 0, 0, 0, 3057, 0, 0, 38, 341, 0, 1, 0, 1, 518, 0, 24, 2, 22242],
          [1, 6248, 0, 0, 4845, 297, 2195, 16251, 0, 0, 122, 0, 0, 0, 151, 3854, 57, 16, 117, 1952],
          [0, 268, 50058, 82, 0, 0, 0, 0, 18299, 0, 72, 176, 1919, 0, 4, 0, 0, 0, 750, 0],
          [2, 0, 0, 0, 324, 3918, 430, 398, 0, 0, 4597, 340, 0, 55597, 0, 88, 0, 7, 0, 0],
          [28, 3664, 0, 60, 0, 0, 2798, 0, 31854, 11338, 0, 0, 0, 51089, 0, 0, 32, 0, 2785, 2],
          [10152, 423, 10, 30671, 32, 1, 10, 0, 2, 453, 10, 0, 92, 1149, 0, 0, 0, 6622, 427, 0],
          [955, 477, 0, 0, 0, 0, 1, 0, 0, 1, 1230, 1, 0, 2, 0, 0, 311, 0, 0, 319],
          [1, 59559, 534, 2547, 0, 48114, 41993, 0, 16916, 0, 0, 0, 2, 5583, 0, 106, 40778, 13, 0, 0],
          [1, 0, 0, 211, 12, 102, 15831, 0, 26, 19, 0, 0, 129, 3, 0, 524, 0, 0, 4, 0]]

def compute_cost(solution: Iterable[int]) -> int:
    """Return objective cost for a permutation solution."""
    sol_list = list(solution)
    cost = 0
    for i in range(SIZE):
        for j in range(SIZE):
            cost += distance[i][j] * flow[sol_list[i]][sol_list[j]]
    return int(cost)


def swap_moves(solution: np.ndarray) -> np.ndarray:
    """Generate all swap neighbors for the given permutation."""
    neighbors = np.zeros((NEIGHBORHOOD_SIZE, SIZE), dtype=int)
    idx = 0
    for i in range(SIZE):
        for j in range(i + 1, SIZE):
            solution[i], solution[j] = solution[j], solution[i]
            neighbors[idx] = solution
            solution[i], solution[j] = solution[j], solution[i]
            idx += 1
    return neighbors


def run_search(seed: int, output: mp.Queue, task_id: int) -> None:
    """Execute hill climbing for one worker and push the result to a queue."""
    rng = np.random.default_rng(seed)
    iteration = 0
    current_solution = rng.permutation(SIZE)
    current_cost = best_cost = compute_cost(current_solution)
    best_solution = current_solution.copy()
    print(f"task {task_id} seed {seed} initial cost {best_cost}")

    while iteration < MAX_ITERS:
        neighbors = swap_moves(current_solution.copy())
        neighbor_costs = np.array([compute_cost(neighbor) for neighbor in neighbors])
        ranked_indices = np.argsort(neighbor_costs)

        improved = False
        for idx in ranked_indices:
            candidate_cost = neighbor_costs[idx]
            if candidate_cost < best_cost:
                current_solution = neighbors[idx].copy()
                current_cost = candidate_cost
                best_solution = current_solution.copy()
                best_cost = candidate_cost
                improved = True
                break

        if not improved:
            print(f"task {task_id} local minimum at iter {iteration} cost {best_cost}")
            break

        iteration += 1

    print(f"task {task_id} best cost {best_cost} iterations {iteration}")
    output.put((task_id, best_solution.tolist(), best_cost, iteration))


def prompt_seed() -> int:
    """Request a seed from stdin, defaulting to the current time."""
    try:
        raw = input("Type your seed (none for using current time as seed): ").strip()
    except EOFError:
        raw = ""
    return int(raw) if raw else int(time.time())


def main() -> None:
    """Spawn parallel hill-climbing workers and report their best results."""
    worker_count = mp.cpu_count()
    print(f"Number of processors: {worker_count}")

    base_seed = prompt_seed()
    base_rng = np.random.default_rng(base_seed)
    worker_seeds = base_rng.integers(0, 1_000_000, size=worker_count)

    output: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=run_search, args=(int(worker_seeds[idx]), output, idx))
        for idx in range(worker_count)
    ]

    start = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    elapsed = time.time() - start
    print(f"time: {elapsed:.4f} s\n")

    results: List[Tuple[int, List[int], int, int]] = [output.get() for _ in processes]
    results.sort(key=lambda item: item[0])

    print("End parallel execution, results:")
    for task_id, best_solution, best_cost, iterations in results:
        print(f"task {task_id}: cost={best_cost} iters={iterations} best={best_solution}")


if __name__ == "__main__":
    main()
    
