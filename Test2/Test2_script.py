import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.plot(range(10),range(10))

fig.savefig('Test2_fig.png', dpi=150)