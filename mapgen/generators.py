
from .noise import *
from .maths import *
from .utils import time_it

@time_it
def archipelago():
    s1 = normalize(domain_warp(offsets=768)) * 3
    s2 = normalize(perlin_square())
    s3 = normalize(gauss_like())
    result =  normalize(normalize(((s1 * 2) + s2)) * s3)
    return result

