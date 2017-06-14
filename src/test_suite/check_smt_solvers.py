import pysmt
import pysmt.factory
from  pysmt.shortcuts import get_env

print("Pysmt file: %s" % pysmt.__file__)
print("Factory module: %s" % pysmt.factory.__file__)
print("Available solvers: %s" % get_env().factory.all_solvers())

