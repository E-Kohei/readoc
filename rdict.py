import re
from functools import lru_cache

class rdict(dict):
    '''Regex dictionary.
    
    This dictionary can take a regex pattern as a search key.
    '''
    def __getitem__(self, key):
        if isinstance(key, re.Pattern):
            pattern = key
            r = [v for k,v in self.items() if rdict._key_search(pattern, k)]
            if len(r) > 0:
                return r
            else:
                raise KeyError(key)
        else:
            return super().__getitem__(key)
    
    def rget(self, *keys):
        patterns = [rdict._key_compile(key) for key in keys]
        r = [v for k,v in self.items() 
             if all([rdict._key_search(p, k) for p in patterns])]
        if len(r) > 0:
            return r
        else:
            raise KeyError(keys)
    
    def __contains__(self, key):
        if isinstance(key, re.Pattern):
            pattern = key
            m = [rdict._key_search(pattern, k) for k in self.keys()]
            return True if any(m) else False
        else:
            return super().__contains__(key)
    
    def is_rin(self, *keys):
        patterns = [rdict._key_compile(key) for key in keys]
        m = [all(rdict._key_search(p, k) for p in patterns) 
             for k in self.keys()]
        return True if any(m) else False
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _key_compile(key):
        return re.compile(key)
    
    @staticmethod
    def _key_search(pattern, key):
        if isinstance(key, str):
            return re.search(pattern, key)
        elif isinstance(key, tuple):
            m = [pattern.search(k) for k in key]
            return m if any(m) else None