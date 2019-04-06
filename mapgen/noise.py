import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product, count
from matplotlib.colors import LinearSegmentedColormap
import noise
from .maths import *



def fbm(shape, p, lower=-np.inf, upper=np.inf):
    # Fourier-based power law noise with frequency bounds.

    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    envelope = (np.power(freq_radial, p, where=freq_radial != 0) *
                (freq_radial > lower) * (freq_radial < upper))
    envelope[0][0] = 0.0
    phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
    return normalize(
        np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope)))


def domain_warp(shape=(1024, 1024), offsets=1200):
    shape = (shape[0], ) * 2

    values = fbm(shape, -2, lower=2.0)
    offsets = offsets * (
        fbm(shape, -2, lower=1.5) + 1j * fbm(shape, -2, lower=1.5))
    result = sample(values, offsets)
    return result


def perlin_square(scale=100.0,
                  octaves=6,
                  persistence=0.5,
                  lacunarity=2.0,
                  shape=(1024, 1024),
                  bounds=(0, 1)):
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=shape[1],
                repeaty=shape[0],
                base=0)
    return normalize(world, bounds)


def white_noise(mapH: int = 512, mapW: int = 512, scale: float = .0001):
    return np.random.random((mapH, mapW)) * scale

