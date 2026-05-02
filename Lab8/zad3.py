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

start_pos = (1,1)
end_pos = (10,10)
moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# ACO
ants = 100
iterations = 200
evaporation_rate = 0.1
pheromone_evaporation_rate=0.1
pheromone_constant=100.0

pheromones = np.ones_like(lab, dtype=float)
solution = None

for i in range(iterations):
    paths = []
    
    for _ in range(ants):
        current_pos = start_pos
        path = [current_pos]
        visited = set([current_pos])
        
        finished = False
        
        # ma maksymalnie 100 ruchów
        for step in range(100):
            if current_pos == end_pos:
                finished = True
                break
        
            y, x = current_pos
            possible_moves = []
            pheromones_levels = []
            
            for y_, x_ in moves:
                new_y, new_x = y + y_, x + x_
                if lab[new_y][new_x] == 0 and (new_y, new_x) not in visited:
                    possible_moves.append((new_y, new_x))
                    pheromones_levels.append(pheromones[new_y][new_x])
            if not possible_moves:
                break
            
            probs = [ p/sum(pheromones_levels) for p in pheromones_levels]
            next_idx = np.random.choice(len(possible_moves), p=probs)
            
            current_pos = possible_moves[next_idx]
            
            path.append(current_pos)
            visited.add(current_pos)
    
        if finished:
            paths.append(path)
            if solution is None or len(path) < len(solution):
                solution = path
                
    pheromones *= (1.0 - evaporation_rate)
    
    for p in paths:
        deposit = pheromone_constant / len(p)
        for y, x in p:
            pheromones[y][x] += deposit
            

if solution:
    print(f"Najlepsza ścieżka: ", solution)
    print(f"Długość: {len(solution)}")
else:
    print("Nie znaleziono ściezki")
    
# Najlepsza ścieżka:  [(1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (1, 5), (1, 6), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (6, 8), (6, 9), (7, 9), (7, 10), (8, 10), (9, 10), (10, 10)]
# Długość: 21
# działa