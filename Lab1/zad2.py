import math
import random
import matplotlib.pyplot as plt
import numpy as np

start = 0.0
limit_a = 50
limit_b = 350
g = 9.81
h = 100.0
v0 = 50.0
err = 5.0

destroyed = False


def drawPlot(rad, landed, target):
    xPoints = np.linspace(start, landed, 500)
    yPoints = -(g / (2 * v0**2 * math.cos(rad)**2)) * xPoints**2 + math.tan(rad) * xPoints + h
    
    plt.figure(figsize=[10, 5])
    plt.plot(xPoints, yPoints)
    plt.title("Trajektoria lotu pocisku")
    plt.xlabel("Odległość (m)")
    plt.ylabel("Wysokość (m)")
    plt.axvline(target, c='red', linestyle='dashed')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    # plt.show()

target = random.randint(limit_a, limit_b)
print(f"Odległość do celu: {target}m")

while not destroyed:
    print("\n====================")
    try:
        alfa = float(input(f"Cel: {target}m\nPodaj kąt strzału (0.0, 90.0): "))
        if alfa <= 0 or alfa >= 90.0:
            raise ValueError
        
    except ValueError:
        print("Wprowadzono niepoprawną liczbę. Spróbuj ponownie")
        continue
    rad = math.radians(alfa)
    
    d = (v0 * math.sin(rad) + math.sqrt(v0**2.0 * math.sin(rad)**2.0 + 2.0 * g * h)) * (v0 * math.cos(rad))/g 
    print(f"Pocisk wylądował na {d:.2f} metrze.")
    
    if abs(target - d) <= 5.0:
        destroyed = True
        print("Cel trafiony!")
        
        drawPlot(rad, d, target)
        plt.savefig('trajektoria.png')
    else:
        print("Pocisk chybił.")







"""
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Zmiana na wielkie litery dla stałych (zgodnie z PEP 8)
START = 0.0
LIMIT_A = 50
LIMIT_B = 340  # Poprawiono z 350 na 340
G = 9.81
H = 100.0
V0 = 50.0
ERR = 5.0

def draw_plot(rad, landed):
    x_points = np.linspace(START, landed, 500)
    y_points = -(G / (2 * V0**2 * math.cos(rad)**2)) * x_points**2 + math.tan(rad) * x_points + H
    
    plt.figure(figsize=[10, 5])
    # Domyślny kolor to niebieski, więc wystarczy zwykły plot
    plt.plot(x_points, y_points)
    plt.title("Trajektoria lotu pocisku")
    plt.xlabel("Odległość (m)")
    plt.ylabel("Wysokość (m)")
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig('trajektoria.png')

target = random.randint(LIMIT_A, LIMIT_B)
print(f"Odległość do celu: {target}m")

destroyed = False
proby = 0  # Dodano licznik prób

while not destroyed:
    proby += 1  # Zwiększanie licznika przy każdej iteracji
    print("\n====================")
    try:
        alfa = float(input(f"Cel: {target}m\nPodaj kąt strzału (0.0, 90.0): "))
        if alfa <= 0 or alfa >= 90.0:
            raise ValueError
        
    except ValueError:
        print("Wprowadzono niepoprawną liczbę. Spróbuj ponownie")
        continue
        
    rad = math.radians(alfa)
    
    # Rozbicie wzoru na dwie linijki dla lepszej czytelności (opcjonalne)
    element_pierwiastka = math.sqrt(V0**2.0 * math.sin(rad)**2.0 + 2.0 * G * H)
    d = (V0 * math.sin(rad) + element_pierwiastka) * (V0 * math.cos(rad)) / G 
    
    print(f"Pocisk wylądował na {d:.2f} metrze.")
    
    if abs(target - d) <= ERR:
        destroyed = True
        print(f"Cel trafiony! Zniszczyłeś zamek po {proby} próbach.")
        draw_plot(rad, d)  # Wykres rysuje się tylko po udanym strzale
    else:
        print("Pocisk chybił.")
"""