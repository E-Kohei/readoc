import numpy as np

from copy import deepcopy
from inspect import signature
from functools import wraps
from collections.abc import Hashable


class DocumentCache():
    '''Document cache data'''
    def __new__(cls):
        # Create singleton instance
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "_cache_data"):
            self._cache_data = {}
    
    @classmethod
    def get_instance(cls):
        return cls()
    
    def __contains__(self, key: Hashable) -> bool:
        return key in self._cache_data
    
    def __len__(self):
        return len(self._cache_data)
    
    def get(self, key: Hashable) -> any:
        return deepcopy(self._cache_data[key])
    
    def set(self, key: Hashable, value: any):
        self._cache_data[key] = deepcopy(value)

    def clear(self):
        self._cache_data.clear()

    def info(self):
        s = cache_length = f"length: {len(self._cache_data)}"
        s += '\n' + '-' * len(cache_length) + '\n'
        for k,v in self._cache_data.items():
            s += f"{k} -> {v}\n"
        return s


def document_cache(f):
    '''Cache the processing result of function.

    This decorator can be used to cache the return value of a function which takes
    'general' non-hashable values (e.g. list and dict) for parameters.
    '''
    @wraps(f)
    def wrapper(document: any, *args, **kwargs):
        ba = signature(f).bind(document, *args, **kwargs)
        ba.apply_defaults()
        sigkey = sorted( convert_hashable(ba.arguments) )
        key = tuple((f.__name__, *sigkey))
        doc_cache = DocumentCache.get_instance()
        if key in doc_cache:
            return doc_cache.get(key)
        else:
            result = f(document, *args, **kwargs)
            doc_cache.set(key, result)
            return result
    return wrapper



def convert_hashable(obj):
    if isinstance(obj, Hashable):
        return obj
    if isinstance(obj, list):
        hashable_seq = []
        for item in obj:
            hashable_seq.append(convert_hashable(item))
        return tuple(hashable_seq)
    if isinstance(obj, dict):
        hashable_map = {}
        for k,v in obj.items():
            hashable_map[k] = convert_hashable(v)
        return tuple(hashable_map.items())
    if isinstance(obj, np.ndarray):
        return hash(obj.tobytes())
    else:
        raise TypeError(f"Cannot convert to hashable: '{obj.__class__}'")
