"""Parallel Tabu Search implementation for the QAP Tai20b instance."""
import argparse
import multiprocessing as mp
import time
from typing import Sequence

import numpy as np

MAX_ITERS = 1000
SIZE = 20
NEIGHBORHOOD_SIZE = SIZE * (SIZE - 1) // 2
TARGET_COST = 122_455_319
SWAP_PAIRS = np.array([(i, j) for i in range(SIZE) for j in range(i + 1, SIZE)], dtype=int)

# Try seed = 5 

# Tai20b 20 122455319 (OPT) (8,16,14,17,4,11,3,19,7,9,1,15,6,13,10,2,5,20,18,12)
# zero-base:                (7,15,13,16,3,10,2,18,6,8,0,14,5,12,9 ,1,4,19,17,11)
distance= [[0,23,29,36,46,53,32,28,64,13,65,36,48,8,47,27,761,812,795,795],
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

flow = [[0, 1341, 283, 17514, 0, 5387, 10, 0, 0, 0, 17307, 98, 122, 1325, 0, 0, 378, 239, 1, 1],
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

def compute_cost(solution: Sequence[int]) -> int:
    """Return objective value for a permutation solution."""
    cost = 0
    for i in range(SIZE):
        for j in range(SIZE):
            cost += distance[i][j] * flow[solution[i]][solution[j]]
    return int(cost)


def delta_swap(solution: Sequence[int], i: int, j: int) -> int:
    """Compute the incremental cost of swapping positions i and j in the permutation."""
    if i == j:
        return 0
    si = solution[i]
    sj = solution[j]
    delta = 0
    for k in range(SIZE):
        if k == i or k == j:
            continue
        sk = solution[k]
        delta += (distance[i][k] - distance[j][k]) * (flow[sj][sk] - flow[si][sk])
        delta += (distance[k][i] - distance[k][j]) * (flow[sk][sj] - flow[sk][si])
    delta += (distance[i][j] - distance[j][i]) * (flow[sj][si] - flow[si][sj])
    return int(delta)


def tabu_duration(rng: np.random.Generator) -> int:
    # Bias toward shorter tenures to encourage exploration but never zero.
    return max(1, int(round((rng.random() ** 3) * 8 * SIZE)))


def run_search(
    seed: int,
    output: mp.Queue,
    task_id: int,
    termination_flag: mp.Value,
    max_iters: int = MAX_ITERS,
    target_cost: int = TARGET_COST,
) -> None:
    """Worker loop: performs Tabu Search and reports the best solution found."""
    print(f"task id {task_id} seed {seed}")
    rng = np.random.default_rng(seed)

    iteration = 0
    current_solution = rng.permutation(SIZE)
    current_cost = best_cost = compute_cost(current_solution)
    best_solution = current_solution.copy()
    print(f"task id {task_id} Initial: {current_solution} cost {best_cost}")

    tabu_list = np.full((SIZE, SIZE), -1, dtype=int)

    while iteration < max_iters and current_cost > target_cost and termination_flag.value == 0:
        neighbor_costs = np.zeros(len(SWAP_PAIRS), dtype=int)

        for idx, (i, j) in enumerate(SWAP_PAIRS):
            neighbor_costs[idx] = current_cost + delta_swap(current_solution, i, j)

        best_move_cost = None
        best_indices = []
        for index in np.argsort(neighbor_costs):
            candidate_cost = neighbor_costs[index]
            if best_move_cost is not None and candidate_cost > best_move_cost:
                break
            swap_i, swap_j = SWAP_PAIRS[index]
            val_i = current_solution[swap_i]
            val_j = current_solution[swap_j]
            tabu_move = tabu_list[swap_i][val_j]
            tabu_reverse = tabu_list[swap_j][val_i]
            is_tabu = tabu_move >= iteration or tabu_reverse >= iteration

            if not is_tabu or candidate_cost < best_cost:
                if best_move_cost is None:
                    best_move_cost = candidate_cost
                if candidate_cost == best_move_cost:
                    best_indices.append(index)

        if best_indices:
            index = rng.choice(best_indices)
            swap_i, swap_j = SWAP_PAIRS[index]
            val_i = current_solution[swap_i]
            val_j = current_solution[swap_j]
            candidate_cost = neighbor_costs[index]
            tabu_move = tabu_list[swap_i][val_j]
            tabu_reverse = tabu_list[swap_j][val_i]
            is_tabu = tabu_move >= iteration or tabu_reverse >= iteration
            if (not is_tabu) or candidate_cost < best_cost:
                current_solution[swap_i], current_solution[swap_j] = current_solution[swap_j], current_solution[swap_i]
                current_cost = candidate_cost
                if candidate_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = candidate_cost
                    print(f"task id {task_id} Best so far {best_solution} cost {best_cost} iter {iteration}")

                tabu_list[swap_i][val_j] = iteration + tabu_duration(rng)
                tabu_list[swap_j][val_i] = iteration + tabu_duration(rng)

        iteration += 1
        if iteration % 100 == 0:
            print(f"task id {task_id} iter {iteration} cost {current_cost}")

    print(f"task id {task_id} Finished with cost {best_cost} at iter {iteration}")
    if best_cost <= target_cost:
        termination_flag.value = 1

    output.put(
        {
            "task_id": task_id,
            "best_solution": best_solution.tolist(),
            "best_cost": int(best_cost),
            "iterations": iteration,
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for seeds, process count, iterations, and target cost."""
    parser = argparse.ArgumentParser(description="Parallel Tabu Search for Tai20b.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Base seed for generating worker seeds (default: current time).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="Number of worker processes to spawn (default: CPU count).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=MAX_ITERS,
        help=f"Maximum iterations per worker (default: {MAX_ITERS}).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_COST,
        help=f"Target cost to stop search early (default: {TARGET_COST}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    process_count = args.processes or mp.cpu_count()
    print(f"Number of processors: {process_count}")

    base_seed = args.seed if args.seed is not None else int(time.time())
    base_rng = np.random.default_rng(base_seed)
    print(f"Base seed: {base_seed}")

    output: mp.Queue = mp.Queue()
    termination_flag = mp.Value("i", 0)
    termination_flag.value = 0

    processes = [
        mp.Process(
            target=run_search,
            args=(
                int(base_rng.integers(0, 1_000_000)),
                output,
                task_id,
                termination_flag,
                args.max_iters,
                args.target,
            ),
        )
        for task_id in range(process_count)
    ]

    start = time.time()
    for process in processes:
        process.start()

    for process in processes:
        process.join()

    results = [output.get() for _ in processes]
    best_run = min(results, key=lambda res: res["best_cost"])

    elapsed = time.time() - start
    print(f"time: {elapsed:.4f} s\n")

    print("End parallel execution, results:")
    for result in sorted(results, key=lambda res: res["task_id"]):
        print(
            f"task {result['task_id']}: cost={result['best_cost']} "
            f"iters={result['iterations']} best={result['best_solution']}"
        )

    print(
        f"\nBest run task {best_run['task_id']} cost={best_run['best_cost']} "
        f"iters={best_run['iterations']}"
    )


if __name__ == "__main__":
    main()
  
