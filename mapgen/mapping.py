from mapgen.maths import normalize, gauss_like
from matplotlib import pyplot as plt
import numpy as np

N = (0, -1)
E = (1, 0)
S = (0, 1)
W = (-1, 0)
C = (0, 0)

vnn_base = np.array([N, E, C, S, W])[:, ::-1]  #(n, (x,y)) -> (n, (y,x))


class Base(dict):
    def __getattr__(self, item):
        return self[item]

    def __dir__(self):
        return super().__dir__() + [str(k) for k in self.keys()]


class Map(Base):
    def __init__(self, shape=(128, 128), parent=None):
        super().__init__()
        self.m1 = np.zeros(shape)
        self.parent = parent

    @property
    def h(self):
        return self.m1.shape[0]

    @property
    def w(self):
        return self.m1.shape[1]

    @property
    def shape(self):
        return self.m1.shape

    def set_map(self, m: np.ndarray):
        self.m1 = m

    def __repr__(self):
        return f"instance of {self.__class__.__name__}: {self.shape}"

    def height_at(self, y, x):
        if self.parent:
            return self.parent.height_at(y, x)
        else:
            return self.m1[int(y), int(x)]

    def show(self, ax=None, figsize=(10, 10)):
        if not ax:
            f, ax = plt.subplots(1, 1, figsize=figsize)
        ax.matshow(self.m1)


def slope(a, center):
    a = a - center + np.random.random() * .01
    before = a.shape
    a = a.flatten()
    s = a.sum()
    a = a / s
    a = a.reshape(before)
    return a


class HeightMap(Map):
    def __init__(self, shape=(128, 128), parent=None):
        Map.__init__(self, shape)
        self.parent = parent
        self.set_map(normalize(gauss_like(shape=shape)))

    def neighbors_downhill(self, y, x):
        #print('finding neighbor:', y, x)
        kernel = vnn_base
        point = np.array([y, x]).astype(int)
        height_here = self.height_at(y, x)
        neighbors = []
        elevations = []
        for q in range(kernel.shape[0]):
            o = kernel[q] + point
            jy, jx = o
            if 0 <= jy < self.h:
                if 0 <= jx < self.w:
                    height_there = self.height_at(jy, jx)
                    #if height_there <= height_here:
                    ##    neighbors.append(o)

                    elevations.append(height_there)
                    neighbors.append(o)

        elevations = np.array(elevations)
        ratios = slope(height_here, elevations)

        return height_here, np.array(neighbors), ratios
        #m2[jy, jx] += spread


class SedimentMap(Map):
    def __init__(self, shape=(128, 128), parent=None):
        Map.__init__(self, shape)
        self.parent = parent


class WaterMap(Map):
    def __init__(self, shape=(128, 128), parent=None):
        Map.__init__(self, shape)
        self.parent = parent

    def evaporate(self, max_rate=.01):
        therm = normalize(
            np.random.random(self.m1.shape), bounds=(.0, max_rate))
        self.m1 -= therm
        self.m1[self.m1 < 0] = 0.

    def add_droplets(self, n=10, mn=.8, mx=1.):
        ix = np.random.random_integers(0, self.w - 1, size=(n, ))
        iy = np.random.random_integers(0, self.h - 1, size=(n, ))
        self.m1[iy, ix] = normalize(np.random.rand(len(iy)), bounds=(mn, mx))

    def diffuse(self, kernel=vnn_base, fluid_transfer_rate=.2):
        batch = np.array(self.m1.nonzero()).T
        m2 = np.zeros_like(self.m1)
        #dN = kernel.shape[0]
        for i in range(batch.shape[0]):
            iy, ix = batch[i]
            water_here = self.m1[iy, ix]
            taking = float(water_here * fluid_transfer_rate)

            self.m1[iy, ix] -= taking

            height_here, neighbors, ratios = self.parent[
                'HeightMap'].neighbors_downhill(iy, ix)

            #print('neighbors:', neighbors)

            dN = neighbors.shape[0]
            #spread = taking / dN
            #m2[iy, ix] += spread

            took = 0
            #print('neighbors:', dN)
            for q in range(neighbors.shape[0]):
                o = neighbors[q] + batch[i]
                jy, jx = o
                if 0 <= jy < self.h:
                    if 0 <= jx < self.w:
                        vx = taking * ratios[q]
                        took += vx

                        m2[jy, jx] += took
            assert (took < taking, f"took:{took}, taking:{taking}")
        self.m1 = m2

    def loop(self,
             rain_chance=.8,
             kernel=vnn_base,
             fluid_transfer_rate=.1,
             rain_density=.6,
             n=16):
        nD = int(max(self.h, self.w) * rain_density)
        for i in range(n):
            if np.random.random() > 1 - rain_chance:
                print(i, f'adding {nD*nD} drops')
                self.add_droplets(nD * nD)
            self.diffuse(
                kernel=kernel, fluid_transfer_rate=fluid_transfer_rate)
            #self.evaporate()
            yield self.m1


maps = [HeightMap, SedimentMap, WaterMap]


class World(Map):
    def __init__(self, shape=(17, 17)):
        Map.__init__(self, shape)
        self.layers = []
        for m in maps:
            k = m.__name__
            self[k] = m(shape=self.shape, parent=self)
            self.layers.append(k)

    def show(self, layers=[], figsize=(10, 10)):
        if not layers:
            layers = self.keys()
        layers = list(layers)
        f, axes = plt.subplots(1, len(layers), figsize=figsize)
        for i, ax in enumerate(axes):
            layer_name = layers[i]
            layer = self[layer_name]
            ax.matshow(layer.m1)
            ax.set_title(layer_name)

    def height_at(self, y, x):
        return sum([self[k].m1[y, x] for k in self.keys()])


# W = World()
# water_cycle = W.WaterMap.loop()
