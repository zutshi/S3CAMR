import modelspec


class PWA(modelspec.ModelGeneric):
    pass


def compute_part_id(guard):
    return


# def sub_model_helper(A, b, C, d, e=None):
#     model = DiscreteMap(A, b, e)
#     partition = Partition(C, d)
#     return SubModel(partition, model)


SubModel = modelspec.PartitionedDiscreteAffineModel


Partition = modelspec.Partition

DiscreteAffineMap = modelspec.DiscreteAffineMap
