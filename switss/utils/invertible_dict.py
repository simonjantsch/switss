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
        """
        :param d: Original dictionary
        :type d: Dict
        :param is_default: If True, will return the empty set if an item is not in the dictionary, defaults to False
        :type is_default: bool, optional
        """
        # validity checks
        assert isinstance(d, dict), "d must be a dictionary but is of type %s" % type(d)
        assert isinstance(is_default, bool), "is_default must be a bool but is of type %s" % type(is_default)
        for k,v in d.items():
            assert isinstance(v,set), "each value must be a set, but value at key=%s is of type %s" % (k,type(v))

        self.__d = d.copy()
        self.__i = None
        self.__is_default = is_default

    def __getitem__(self, key):
        if self.__is_default and key not in self.__d:
            return set()
        return self.__d[key]

    @property
    def is_default(self):
        return self.__is_default

    def add(self,key,item):
        """Adds an item to a key-mapping."""
        if key not in self.__d:
            self.__d[key] = set()
        self.__d[key].add(item)
        
        if self.__i is not None:
            if item not in self.__i:
                self.__i.__d[item] = set()
            self.__i.__d[item].add(key)

    def remove(self,key,item):
        """Removes an item from a key-mapping"""
        assert key in self.__d, "Key %s not in dictionary" % key
        assert item in self.__d[key], "Item not at dict[%s]" % key

        self.__d[key].remove(item)

        if self.__i is not None:
            self.__i.__d[item].remove(key)


    def __contains__(self, key):
        return key in self.__d

    def __repr__(self):
        return self.__d.__repr__()

    def keys(self):
        """The keys of the underlying dictionary.

        :return: Keys.
        :rtype: Iterable
        """        
        return self.__d.keys()

    def items(self):
        """The items of the underlying dictionary.

        :return: iterable of key-value pairs.
        :rtype: Iterable
        """        
        return self.__d.items()

    @property
    def inv(self):
        """
        Implementation of the inverse mapping from values to sets of keys, i.e.

        .. math::

            \\text{inv}: V \mapsto 2^K,\quad \\text{inv}(v) = \{k \in K \mid v \in f(k) \}

        :return: The respective dictionary.
        :rtype: InvertibleDict
        """        
        if self.__i is None:
            i = defaultdict(set)
            for key, vals in self.__d.items():
                for val in vals:
                    i[val].add(key)
            self.__i = InvertibleDict(dict(i), is_default=self.__is_default)
            self.__i.__i = self
        return self.__i
