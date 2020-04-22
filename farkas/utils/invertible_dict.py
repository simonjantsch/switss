from collections import defaultdict

class InvertibleDict:
    """An InvertibleDict is a dictionary that is invertible. It implements a mapping :math:`f` 
    from keys to sets of values

    .. math::

        f: K \mapsto 2^V,
    
    but also supports the reverse direction, i.e.
    
    .. math::

        \\text{inv}: V \mapsto 2^K,\quad \\text{inv}(v) = \{k \in K \mid v \in f(k) \},

    for each value to its respective set of keys.

    """    
    def __init__(self, d, is_default=False):
        self.d = d.copy()
        self.i = None
        self.is_default = is_default

    def __getitem__(self, key):
        if self.is_default and key not in self.d:
            return set()
        return self.d[key]

    def __setitem__(self,key,item):
        if key not in self.d:
            self.d[key] = set()
        self.d[key].add(item)
        
        if self.i is not None:
            if item not in self.i:
                self.i[item] = set()
            self.i[item].add(key)

    def __contains__(self, item):
        return item in self.d

    def __repr__(self):
        return self.d.__repr__()

    def keys(self):
        """The keys of the underlying dictionary.

        :return: Keys.
        :rtype: Iterable
        """        
        return self.d.keys()

    def items(self):
        """The items of the underlying dictionary.

        :return: iterable of key-value pairs.
        :rtype: Iterable
        """        
        return self.d.items()

    @property
    def inv(self):
        """
        Implementation of the inverse mapping from values to sets of keys, i.e.

        .. math::

            \\text{inv}: V \mapsto 2^K,\quad \\text{inv}(v) = \{k \in K \mid v \in f(k) \}

        :return: The respective dictionary.
        :rtype: defaultdict(set)
        """        
        if self.i is None:
            i = defaultdict(set)
            for key, vals in self.d.items():
                for val in vals:
                    i[val].add(key)
            self.i = InvertibleDict(dict(i), is_default=self.is_default)
        return self.i
