import pygad
import numpy
import time

items = ['zegar', 'obraz-pejzaz', 'obraz-portret', 'radio', 'laptop',
         'lampa nocna', 'srebrne sztucce', 'porcelana', 'figura z brazu', 
         'skorzana torebka', 'odkurzacz']

values = numpy.array([100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300])
weights = numpy.array([7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15])

max_weight = 25
max_profit = numpy.sum(values)


#definiujemy funkcję fitness
def fitness_func(model, solution, solution_idx): # solution - wektor np.(1, 1, 0, 0, ...)
    weight = numpy.sum( solution * weights )
    profit = numpy.sum( solution * values )
    
    if weight > max_weight:
        return 0

    return profit

#definiujemy parametry chromosomu
#geny to liczby: 0 (nie kradnie) lub 1 (kradnie)
gene_space = [0, 1]

fitness_function = fitness_func

sol_per_pop = 40 # chromosomów w populacji
num_genes = len(items) # genów w chromosomie

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
mutation_percent_genes = 10

# e + f)
trials_num = 10
success = 0
trials_times = []
target = 1630

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
for i in range(trials_num):
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
    #  stop_criteria=[f"reach_{target}"] # Zatrzymuje algorytm po znalezieniu 1630
    
    start = time.time()
    #uruchomienie algorytmu
    ga_instance.run()
    stop = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if solution_fitness == target:
        success += 1
        trials_times.append(round(stop-start,6))

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

print("Najlepsza możliwość:")
predicted_profit = numpy.sum(values*solution)
print("  - Wartość: ", predicted_profit)
predicted_weight = numpy.sum(weights*solution)
print("  - Waga: ", predicted_weight)

print(f" Na {trials_num} algorytm wyliczył 1630 {success} razy.")
print(trials_times)
print(f"Średni czas działania algorytmu to {numpy.mean(trials_times):.6f}")
#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
