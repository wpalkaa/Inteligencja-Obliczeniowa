import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

def endurance(p):
    
    print(p)
    x = p[:,0]
    y = p[:,1]
    z = p[:,2]
    u = p[:,3]
    v = p[:,4]
    w = p[:,5]
    
    endurance = np.exp(-2 * (y - np.sin(x))**2) + np.sin(z * u) + np.cos(v * w)
    
    return -endurance


# x_max = [2, 2]
# x_min = [1, 1]
x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} # pamięć własna, jak bardzo cząstka dąży do własnego rozwiązania ; wpływ stada ; wsp. bezwładności
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)
optimizer.optimize(endurance, iters=1000)

# Perform optimization
cost, pos = optimizer.optimize(fx.sphere, iters=200)

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history

# Plot!
plot_cost_history(cost_history)
plt.show()