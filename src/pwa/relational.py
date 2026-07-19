from . import modelspec


PWARelational = modelspec.ModelGeneric


class KPath(modelspec.PartitionedDiscreteAffineModel):
    """Generelization of Relational Modeling"""
    def __init__(self, m, p, pnexts, p_future):
        r"""
        The k-length path modeled as:
            p -> pnext -> ... p2 -> pk ==> x' = m(x)
        where p, pnext, p_future enforces the order: [p, pnext, p2..., pk]
        [p2,...,pk] \in p_future.
        Currently, p_future is used only for modeling and not for BMC
        queries.
        """
        # Model's ID is its first partition's ID
        # This is acceptable because even a k-relational submodel is
        # identified by which cell it begins in and nothing else.
        #self.ID = p
        self.p = p
        self.pnexts = pnexts

        # convinience function for now
        self.pnext = self.pnexts[0] if len(pnexts) == 1 else None

        self.p_future = p_future
        self.m = m
        # make p = p1, i.e., check sat against p[0] and ID is p[0]'s ID
        super(self.__class__, self).__init__(p, m)
        return

    def __repr__(self):
        s = f'({self.p},{self.pnexts},{self.p_future},{self.m})'
        return s

    def __str__(self):
        s = f'SubModel ->(\n{self.p},\n{self.pnexts},\n{self.p_future},\n{self.m})'
        return s


class Relation(modelspec.PartitionedDiscreteAffineModel):
    def __init__(self, p1, p2, m):
        """
        The edge between p1 (source parition) and p2 (target parition)
        is modeled as:
            p1 -> p2 ==> x' = m(x)
        """
        self.p1 = p1
        self.p2 = p2
        self.pnext = p2
        self.m = m
        # make p = p1, i.e., check sat against p1 and ID is p1's ID
        super(self.__class__, self).__init__(p1, m)
        return

    def __repr__(self):
        s = f'({self.p1},{self.p2},{self.m})'
        return s

    def __str__(self):
        s = f'SubModel ->(\n{self.p1},\n{self.p2},\n{self.m})'
        return s


Partition = modelspec.Partition

DiscreteAffineMap = modelspec.DiscreteAffineMap
DiscretePolyMap = modelspec.DiscretePolyMap
