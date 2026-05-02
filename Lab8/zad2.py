import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")


COORDS = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(15)]

COORDS = (
 (0, 0),
 (10, 0),
 (20, 0),
 (30, 0),
 (40, 0),
 (0, 10),
 (10, 10),
 (20, 10),
 (30, 10),
 (40, 10),
 (0, 20),
 (10, 20),
 (20, 20),
 (30, 20),
 (40, 20),
 (0, 30),
 (10, 30),
 (20, 30),
 (30, 30),
 (40, 30),
 (0, 40),
 (10, 40),
 (20, 40),
 (30, 40),
 (40, 40),
)

def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()
# alpha - ważność feromonów
# beta - ważność odległości (wybierają tylko to co jest obok czy nie)
colony = AntColony(COORDS, ant_count=300, alpha=0.1, beta=2.0, 
                    pheromone_evaporation_rate=0.8, pheromone_constant=700.0,
                    iterations=300)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()

# T1 (0,5; 1.2): 331.7557993434155 # pheromone_evaporation_rate=0.4, pheromone_constant=1000.0,
# T2 (0.5; 0.1): 467.4577994414623 # pheromone_evaporation_rate=0.4, pheromone_constant=1000.0,
# T3 (0.1; 2.0): 313.3626562684468 # pheromone_evaporation_rate=0.8, pheromone_constant=700.0,
# T4 (3.0; 3.0): 364.69862217974685 # pheromone_evaporation_rate=0.6, pheromone_constant=200.0,
# T5 (0.5; 3.2): 354.29108460991023 # pheromone_evaporation_rate=0.2, pheromone_constant=3000.0,



# d) 306.0555127546399