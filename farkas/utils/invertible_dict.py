from collections import defaultdict

class InvertibleDict:
    def __init__(self, d, is_default=False, default=None):
        self.d = d.copy()
        self.i = None
        self.is_default = is_default
        self.default = default
    
    def __getitem__(self, key):
        if self.is_default and key not in self.d:
            return self.default()
        return self.d[key]
    
    def __contains__(self, item):
        return item in self.d

    def __repr__(self):
        return self.d.__repr__()

    def keys(self):
        return self.d.keys()

    def items(self):
        return self.d.items()

    @property
    def inv(self):
        if self.i is None:
            self.i = defaultdict(set)
            for key, values in self.d.items():
                for val in values:
                    self.i[val].add(key)
            self.i = InvertibleDict(dict(self.i), is_default=self.is_default, default=self.default)
        return self.i