# Parallel-Metaheuristics
# Metaheuristics Course - UdeA

This project contains some implementation examples of parallel metaheuristics in Python.

- example1.py - presents an example of the use of the python random number generator.
- example2.py - implements a parallel execution of a dummy function.
- hill_climbing.py - implements a very simple hill climbing algorithm for solving a QAP problem.
- hill_climbing-tai-20b.py - implements the hill climbing algorithm but solving a QAPLIB instance (Tai20b).
- par_hill_climbing-tai-20b.py - implements a parallel version (independent-multiwalks) of the hill climbing algorithm.
- tabu_search-tai-20b.py - implements a basic tabu search algorithm.
- par_tabu_search-tai-20b.py - implements a parallel version of the tabu search algorithm.
- par_tabu_search-tai-20b-vp3.py - implements a parallel tabu search for python v3.
- par_tabu_search-tai-20b-td.py - implements a parallel tabu search, inluding a termination detection mechanism.

Currently, the implemented metaheuristics are not so efficient, there are many optimization on the code that must be done. 
I am using these examples only to explore how to implement parallelism in python.

Enjoy it!
