import pygad
import numpy as np
import time
import math

lab = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,0,0,0,1,0,1,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,1,0,1,1,0,0,1,1,0,1],
    [1,0,0,1,1,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,1],
    [1,0,1,0,0,1,1,0,1,0,0,1],
    [1,0,1,1,1,0,0,0,1,1,0,1],
    [1,0,1,0,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])

moves_map = {
    0: "up",
    1: "down",
    2: "left",
    3: "right"
}


max_steps = 30
start_position = (1,1)
end_position = (10,10)

#definiujemy funkcję fitness
def fitness_func(model, solution, solution_idx):
    y, x = start_position
    steps = 0
    penalty = 0
    
    for move in solution:
        steps += 1
        
        x_next, y_next = x, y
        if move == 0 : y_next -= 1
        elif move == 1 : y_next += 1
        elif move == 2 : x_next -= 1
        else: x_next += 1
        
        if lab[y_next][x_next] == 0:
            x, y = x_next, y_next
        else:
            penalty += 5
        
        if (y, x) == end_position: # jeżeli meta
            return 1000 - 3*steps - penalty
        
    distance = abs(end_position[0] - y) + abs(end_position[1] - x)
    fit = -distance*5 - penalty
    
    return fit

#definiujemy parametry chromosomu
gene_space = [0, 1, 2, 3]

fitness_function = fitness_func

sol_per_pop = 100 # chromosomów w populacji
num_genes = max_steps # genów w chromosomie

num_parents_mating = 45 # ile rodziców do rozmnażania (około 50% populacji)
num_generations = 300 # ile pokolen
keep_parents = 7 # ilu rodziców zachować (kilka procent)

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 4 #trzeba dać więcej bo inaczej jest warning



success = 0
times = []

for i in range(10):
        
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        stop_criteria=f"reach_100")
    start = time.time()
    ga_instance.run()
    end = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if solution_fitness > 100:
        success += 1
    times.append(round(end-start,6))

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

print([moves_map[m] for m in solution])

print(f"Sukcesy: {success}")
print(f"Czasy: {times}")
print(f"Średni czas: {np.mean(times)}")


#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()


