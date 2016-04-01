import modelspec


PWA = modelspec.ModelGeneric

# def sub_model_helper(A, b, C, d, e=None):
#     model = DiscreteMap(A, b, e)
#     partition = Partition(C, d)
#     return SubModel(partition, model)


SubModel = modelspec.PartitionedDiscreteAffineModel


Partition = modelspec.Partition

DiscreteAffineMap = modelspec.DiscreteAffineMap
