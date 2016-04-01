import modelspec


PWARelational = modelspec.ModelGeneric


class Relation(modelspec.PartitionedDiscreteAffineModel):
    def __init__(self, p1, p2, m):
        """
        The edge between p1 (source parition) and p2 (target parition)
        is modeled as:
            p1 -> p2 ==> x' = m(x)
        """
        self.p1 = p1
        self.p2 = p2
        self.m = m
        # make p = p1, i.e., check sat against p1 and ID is p1's ID
        super(Relation, self).__init__(p1, m)
        return

    def __repr__(self):
        s = '({},{},{})'.format(self.p1, self.p2, self.m)
        return s

    def __str__(self):
        s = 'SubModel ->(\n{},\n{},\n{})'.format(self.p1, self.p2, self.m)
        return s


Partition = modelspec.Partition

DiscreteAffineMap = modelspec.DiscreteAffineMap
