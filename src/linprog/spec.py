import collections

# Tuple contains lp results. It follows the scipy convention
OPTRES = collections.namedtuple('optres', ('fun', 'x', 'status', 'success'))
