"""Compare sequential versus parallel execution for a dummy workload."""
import multiprocessing as mp
import queue
import time
from typing import Iterable, List

import numpy as np

ITERATIONS_PER_WORKER = 1_000_000


def prompt_seed() -> int:
    """Request a seed from stdin, defaulting to the current time."""
    try:
        raw = input("Type your seed (none uses current time): ").strip()
    except EOFError:
        raw = ""
    return int(raw) if raw else int(time.time())


def dummy_func(seed: int, iterations: int, output_queue) -> None:
    """Simulate work and push a result into the provided queue."""
    rng = np.random.default_rng(seed)
    last_val = 0.0
    for i in range(1, iterations):
        last_val = 234.2 / i * rng.integers(100)
    output_queue.put(last_val)


def run_sequential(seeds: Iterable[int], iterations: int) -> List[float]:
    """Execute dummy_func sequentially for each seed and return outputs."""
    fifo_queue: queue.Queue = queue.Queue()
    start = time.time()
    for seed in seeds:
        dummy_func(int(seed), iterations, fifo_queue)
    results = [fifo_queue.get() for _ in seeds]
    duration = time.time() - start
    print(f"Sequential execution time: {duration:.4f} s")
    return results


def run_parallel(seeds: Iterable[int], iterations: int) -> List[float]:
    """Execute dummy_func in parallel for each seed and return outputs."""
    output: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=dummy_func, args=(int(seed), iterations, output))
        for seed in seeds
    ]

    start = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    results = [output.get() for _ in processes]
    duration = time.time() - start
    print(f"Parallel execution time: {duration:.4f} s")
    return results


def main() -> None:
    """Compare sequential versus parallel execution of a dummy workload."""
    worker_count = mp.cpu_count()
    print(f"Number of processors: {worker_count}")

    base_seed = prompt_seed()
    base_rng = np.random.default_rng(base_seed)
    input_vector = base_rng.integers(0, 100, worker_count)

    print(f"Base seed: {base_seed}")
    print(f"Input vector: {input_vector}")

    print("Starting sequential execution")
    sequential_results = run_sequential(input_vector, ITERATIONS_PER_WORKER)
    print(sequential_results)

    print("Starting parallel execution")
    parallel_results = run_parallel(input_vector, ITERATIONS_PER_WORKER)
    print(parallel_results)


if __name__ == "__main__":
    main()
