"""Compare sequential vs parallel execution for a dummy workload."""
import argparse
import multiprocessing as mp
import queue
import time
from typing import List, Sequence, Union

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dummy parallel workload comparison.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Base seed for generating input seeds (default: current time).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="Number of workers to run (default: CPU count).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1_000_000,
        help="Iterations per worker (default: 1_000_000).",
    )
    return parser.parse_args()


def resolve_seed(seed_arg: int | None) -> int:
    try:
        if seed_arg is not None:
            return seed_arg
        raw = input("Type your seed (empty uses the current time): ").strip()
        return int(raw) if raw else int(time.time())
    except EOFError:
        return int(time.time())


def dummy_func(seed: int, iterations: int, output: Union[queue.Queue, mp.queues.Queue]) -> None:
    """Simulate work and emit a final value into the provided queue."""
    rng = np.random.default_rng(seed)
    last_val = 0.0
    for i in range(1, iterations):
        last_val = 234.2 / i * rng.integers(100)
    output.put(last_val)


def run_sequential(seeds: Sequence[int], iterations: int) -> List[float]:
    fifo_queue: queue.Queue = queue.Queue()
    start = time.time()
    for seed in seeds:
        dummy_func(seed, iterations, fifo_queue)
    results = [fifo_queue.get() for _ in seeds]
    duration = time.time() - start
    print(f"Sequential execution time: {duration:.4f} s")
    return results


def run_parallel(seeds: Sequence[int], iterations: int) -> List[float]:
    output: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=dummy_func, args=(seed, iterations, output))
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
    args = parse_args()
    worker_count = args.processes or mp.cpu_count()
    print(f"Number of processors: {worker_count}")

    base_seed = resolve_seed(args.seed)
    base_rng = np.random.default_rng(base_seed)
    print(f"Base seed: {base_seed}")

    input_vector = base_rng.integers(0, 100, worker_count)
    print(f"Input vector: {input_vector}")

    print("Starting sequential execution")
    seq_results = run_sequential(input_vector, args.iterations)
    print(seq_results)

    print("Starting parallel execution")
    par_results = run_parallel(input_vector, args.iterations)
    print(par_results)


if __name__ == "__main__":
    main()
