import numpy as np
import random

max_iters = 100
size = 20
neigh_size = int(size*(size-1)/2)

flow=   [[0, 0, 5,0, 5,2,10,3,1, 5, 5, 5, 0, 0, 5, 4, 4, 0, 0, 1 ],
        [0, 0, 3,10,5,1, 5,1,2, 4, 2, 5, 0,10,10, 3, 0, 5,10, 5 ],
        [5, 3, 0,2, 0,5, 2,4,4, 5, 0, 0, 0, 5, 1, 0, 0, 5, 0, 0 ],
        [0,10, 2,0, 1,0, 5,2,1, 0,10, 2, 2, 0, 2, 1, 5, 2, 5, 5 ],
        [5, 5, 0,1, 0,5, 6,5,2, 5, 2, 0, 5, 1, 1, 1, 5, 2, 5, 1 ],
        [2, 1, 5,0, 5,0, 5,2,1, 6, 0, 0,10, 0, 2, 0, 1, 0, 1, 5 ],
        [10,5, 2,5, 6,5, 0,0,0, 0, 5,10, 2, 2, 5, 1, 2, 1, 0,10 ],
        [3, 1, 4,2, 5,2, 0,0,1, 1,10,10, 2, 0,10, 2, 5, 2, 2,10 ],
        [1, 2, 4,1, 2,1, 0,1,0, 2, 0, 3, 5, 5, 0, 5, 0, 0, 0, 2 ],
        [5,4, 5,0, 5,6, 0,1,2, 0, 5, 5, 0, 5, 1, 0, 0, 5, 5, 2 ],
        [5,2, 0,10,2,0, 5,10,0,5, 0, 5, 2, 5, 1,10, 0, 2, 2, 5 ],
        [5,5, 0,2, 0,0,10,10,3,5, 5, 0, 2,10, 5, 0, 1, 1, 2, 5 ],
        [0,0, 0,2, 5,10,2,2, 5,0, 2, 2, 0, 2, 2, 1, 0, 0, 0, 5 ],
        [0,10,5,0, 1,0, 2,0, 5,5, 5,10, 2, 0, 5, 5, 1, 5, 5, 0 ],
        [5,10,1,2, 1,2, 5,10,0,1, 1, 5, 2, 5, 0, 3, 0, 5,10,10 ],
        [4, 3,0,1, 1,0, 1,2, 5,0,10, 0, 1, 5, 3, 0, 0, 0, 2, 0 ],
        [4, 0,0,5, 5,1, 2,5, 0,0, 0, 1, 0, 1, 0, 0, 0, 5, 2, 0 ],
        [0, 5,5,2, 2,0, 1,2, 0,5, 2, 1, 0, 5, 5, 0, 5, 0, 1, 1 ],
        [0,10,0,5, 5,1, 0,2, 0,5, 2, 2, 0, 5,10, 2, 2, 1, 0, 6 ],
        [1, 5,0,5, 1,5,10,10,2,2, 5, 5, 5, 0,10, 0, 0, 1, 6, 0 ]]

distance=[[0,1,2,3,4,1,2,3,4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7 ],
        [1,0,1,2,3,2,1,2,3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6 ],
        [2,1,0,1,2,3,2,1,2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5 ],
        [3,2,1,0,1,4,3,2,1, 2, 5, 4, 3, 2, 3, 6, 5, 4, 3, 4 ],
        [4,3,2,1,0,5,4,3,2, 1, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3 ],
        [1,2,3,4,5,0,1,2,3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6 ],
        [2,1,2,3,4,1,0,1,2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5 ],
        [3,2,1,2,3,2,1,0,1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4 ],
        [4,3,2,1,2,3,2,1,0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3 ],
        [5,4,3,2,1,4,3,2,1,0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2 ],
        [2,3,4,5,6,1,2,3,4,5, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5 ],
        [3,2,3,4,5,2,1,2,3,4, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4 ],
        [4,3,2,3,4,3,2,1,2,3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3 ],
        [5,4,3,2,3,4,3,2,1,2, 3, 2, 1, 0, 1, 4, 3, 2, 1, 2 ],
        [6,5,4,3,2,5,4,3,2,1, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1 ],
        [3,4,5,6,7,2,3,4,5,6, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4 ],
        [4,3,4,5,6,3,2,3,4,5, 2, 1, 2, 3, 4, 1, 0, 1, 2, 3 ],
        [5,4,3,4,5,4,3,2,3,4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2 ],
        [6,5,4,3,4,5,4,3,2,3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1 ],
        [7,6,5,4,3,6,5,4,3,2, 5, 4, 3, 2, 1, 4, 3, 2, 1, 0 ]]


def compute_cost(sol):
  cost=0
  for i in range(size):
    for j in range(size):
        cost+=distance[i][j] *flow[sol[i]][sol[j]]
  return cost

def swap_move(sol_n):
    neighbors = np.zeros((neigh_size, size), dtype=int)
    idx=0
    for i in range(size):
        j=i+1
        for j in range(size):
            if  i<j:
                sol_n[j], sol_n[i] = sol_n[i], sol_n[j]
                neighbors[idx] = sol_n
                sol_n[i], sol_n[j] = sol_n[j], sol_n[i]
                idx=idx+1
    '''print("neigh:", len(neighbors))
    print(*neighbors, sep="\n")'''
    return neighbors

def run_search():
    global max_iters
    num_iter = 0
    curnt_sol = random.sample(range(size), size)
    best_soln = curnt_sol
    best_cost = curnt_cost = compute_cost(curnt_sol)
    print("Initial: %s cost %s " % (curnt_sol, best_cost))

    # flag to signal the algorithm's termination
    flag_end = False
    while (num_iter < max_iters and (not flag_end)):

        neighbors = swap_move(curnt_sol)  # make a move to neighbors

        # holds the cost of the neighbors
        cost = np.zeros((len(neighbors)))  
        for index in range(len(neighbors)):
            # evaluate the cost of the candidate neighbors
            cost[index] = compute_cost(neighbors[index])  
        rank = np.argsort(cost)  # sorted index based on cost
        print ("ranks: \n", *rank)
        neighbors = neighbors[rank]
        for h in range(5):
            print(cost[h])

        first_best = False
        for j in rank:
            curnt_cost = cost[j]
            if  curnt_cost <  best_cost:
                curnt_sol = best_soln = neighbors[j].tolist()
                best_cost = curnt_cost
                print("Found better sol %s cost: %s " % (best_soln, best_cost))
                first_best = True

            if (not first_best):
                print("Local minimum found!!")
                flag_end = True
                break
        num_iter+=1

    print("Best sol %s cost: %s max_iters= %s" % (best_soln, best_cost , num_iter))

# calling the main function, where the program starts running
if __name__== "__main__":  
    run_search()
