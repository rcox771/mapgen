from mapgen.generators import archipelago
from mapgen.rendering import render
from matplotlib import pyplot as plt
import numpy as np



world = archipelago()
rendered = render(world)

f, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=100)
ax.imshow(rendered)
plt.show()
