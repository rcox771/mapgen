from matplotlib.colors import LinearSegmentedColormap, LightSource, rgb2hex
import matplotlib.cm as cm
import numpy as np
from .maths import lerp

default_colors = [
    {
        'name': 'ocean',
        'color': (0, 100, 190),
        'height': .0
    },
    {
        'name': 'waterDeep',
        'color': (0, 119, 190),
        'height': .3
    },
    {
        'name': 'waterShallow',
        'color': (0, 142, 190),
        'height': .4
    },
    {
        'name': 'sand',
        'color': (228, 226, 154),
        'height': .45
    },
    {
        'name': 'grass',
        'color': (19, 139, 39),
        'height': .48
    },
    {
        'name': 'grassDark',
        'color': (11, 105, 27),
        'height': .55
    },
    {
        'name': 'rockGray',
        'color': (78, 78, 78),
        'height': .66
    },
    {
        'name': 'rock',
        'color': (104, 78, 54),
        'height': .72
    },
    {
        'name': 'rockDark',
        'color': (80, 60, 42),
        'height': .76
    },
    {
        'name': 'snow',
        'color': (250, 250, 250),
        'height': 1.0
    },
]

sc = list(sorted(default_colors, key=lambda k: k['height']))
vals = [(k['height'], tuple(np.array(k['color']) / 255.)) for k in sc]
default_cmap = LinearSegmentedColormap.from_list('world', vals)


def render(a, cmap=default_cmap, land_mask=None, angle=270):
    if land_mask is None: land_mask = np.ones_like(a)
    ls = LightSource(azdeg=angle, altdeg=30)
    land = ls.shade(
        a, cmap=cmap, vert_exag=10.0, blend_mode='overlay')[:, :, :3]
    water = np.tile((0.25, 0.35, 0.55), a.shape + (1, ))
    return lerp(water, land, land_mask[:, :, np.newaxis])



