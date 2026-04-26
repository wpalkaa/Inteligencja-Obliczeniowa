import pygad
import numpy
import time
import math

def endurance(x, y, z, u, v, w): 
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w) 


#definiujemy funkcję fitness
def fitness_func(model, solution, solution_idx): # solution - wektor np.(1, 1, 0, 0, ...)
    return endurance(*solution)

#definiujemy parametry chromosomu
#geny to liczby: 0->1
gene_space = {"low": 0.0, "high": 0.999999}

fitness_function = fitness_func

sol_per_pop = 40 # chromosomów w populacji
num_genes = 6 # genów w chromosomie

num_parents_mating = 20 # ile rodziców do rozmnażania (około 50% populacji)
num_generations = 30 # ile pokolen
keep_parents = 5 # ilu rodziców zachować (kilka procent)

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 30 #trzeba dać więcej bo inaczej jest warning

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
                    mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()





# PS C:\Users\Wojtek\Desktop\Uczelnia\Inteligencja-Obliczeniowa\Lab7> python3 .\zad2.py
# Parameters of the best solution : [0.90152277 0.76930662 0.96905628 0.957854   0.59325463 0.00342579]
# Fitness value of the best solution = 2.8001012681710424
# PS C:\Users\Wojtek\Desktop\Uczelnia\Inteligencja-Obliczeniowa\Lab7> python3 .\zad2.py
# Parameters of the best solution : [0.66337529 0.63150482 0.99893062 0.98131013 0.46813575 0.07069168]
# Fitness value of the best solution = 2.8296006169822028
# PS C:\Users\Wojtek\Desktop\Uczelnia\Inteligencja-Obliczeniowa\Lab7> python3 .\zad2.py
# Parameters of the best solution : [0.61137817 0.56492189 0.99338073 0.98750473 0.11046216 0.02179103]
# Fitness value of the best solution = 2.8308686937102223
# PS C:\Users\Wojtek\Desktop\Uczelnia\Inteligencja-Obliczeniowa\Lab7> python3 .\zad2.py
# Parameters of the best solution : [0.8081684  0.7165398  0.99159458 0.99351176 0.18738292 0.01165805]
# Fitness value of the best solution = 2.8332745742244114

