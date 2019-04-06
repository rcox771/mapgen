from functools import wraps
from time import time


def time_it(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'Elapsed: {round(end-start, 3)}')
        return result
    return wrapper