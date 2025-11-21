# Parallel-Metaheuristics
# Metaheuristics Course - UdeA

This project contains some implementation examples of parallel metaheuristics in Python.

- 1_warmup_example.py - presents an example of the use of the python random number generator.
- 2_first_dummy_parallel.py - implements a parallel execution of a dummy function.
- 3_hill_climbing_seq.py - implements a hill climbing algorithm for solving a QAPLIB instance (Tai20b).
- 4_hill_climbing-par.py - implements a parallel version (independent-multiwalks) of the hill climbing algorithm.
- 5_tabu_search-par.py - implements a parallel tabu search for for solving a QAPLIB instance (Tai20b).
- 6_tabu_search-par-td.py - implements a parallel tabu search, including a termination detection mechanism.

Currently, the implemented metaheuristics are not so efficient, there are some code optimization that must be done. 
These examples are useful to explore how to implement parallelism in python.

Enjoy it!

## Usage

- Install NumPy (`pip install numpy`) if needed.
- Run any script directly, e.g.:
  - `python 1_warmup_example.py --seed 123 --count 5`
  - `python 4_hill_climbing-par.py`
  - `python 5_tabu_search-par.py --processes 4 --max-iters 500 --seed 42`
  - `python 6_tabu_search-par-td.py --processes 4 --target 122455319`
