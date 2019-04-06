import numpy as np

def lerp(x, y, a):
    return (1.0 - a) * x + a * y

def normalize(x, bounds=(0, 1)):
    return np.interp(x, (x.min(), x.max()), bounds)

def gauss_like(shape=(1024, 1024), bounds=(-1, 1)):
    lower, upper = bounds
    x, y = np.meshgrid(
        np.linspace(lower, upper, shape[0]), np.linspace(
            lower, upper, shape[1]))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))
    return g

def sample(a, offset):
    shape = np.array(a.shape)
    delta = np.array((offset.real, offset.imag))
    coords = np.array(np.meshgrid(*map(range, shape))) - delta
    lower_coords = np.floor(coords).astype(int)
    upper_coords = lower_coords + 1
    coord_offsets = coords - lower_coords
    lower_coords %= shape[:, np.newaxis, np.newaxis]
    upper_coords %= shape[:, np.newaxis, np.newaxis]
    result = lerp(
        lerp(a[lower_coords[1], lower_coords[0]],
             a[lower_coords[1], upper_coords[0]], coord_offsets[0]),
        lerp(a[upper_coords[1], lower_coords[0]],
             a[upper_coords[1], upper_coords[0]], coord_offsets[0]),
        coord_offsets[1])
    return result