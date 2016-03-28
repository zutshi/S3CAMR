import numpy as np
import constraints as cons
import itertools


# TODO: change cons to cons_object which has cons in addition to all states
# a concrete state has
def ival_cons_to_cell_list(cons, eps):
    num_dims = cons.dim

    Al = cell_from_concrete(cons.l, eps)
    Ah = cell_from_concrete(cons.h, eps)

    x_range_list = []
    for i in range(num_dims):

        # x_range_list.append(np.arange(cons.l[i], cons.h[i], eps[i]))

        x_range = range(Al[i], Ah[i], 1)

        # make the range inclusive of the last element,  because we want things to be sound

        x_range.append(Ah[i])
        x_range_list.append(x_range)

    # construct a grid
        #grid = np.meshgrid(*x_range_list, copy=False)
        grid = np.meshgrid(*x_range_list)

#         # sometimes fails with memory errors
#         try:
#             grid = np.meshgrid(*x_range_list)
#         except MemoryError as e:
#             print('meshgrid with the below x_cons failed')
#             print(cons)
#             raise e

    # create iterators which iterate over each element of the dim. array

#         x_iter_list = []
#         for i in range(self.num_dims.x):
#             x_iter_list.append(np.nditer(grid[i]))
    x_iter_list = [np.nditer(grid[i]) for i in range(num_dims)]

    return x_iter_list


def get_cells_from_ival_constraints(ic, eps):
    x_list = ival_cons_to_cell_list(ic, eps)
    cells = [tuple(map(int, abs_state_array_repr)) for abs_state_array_repr in zip(*x_list)]
    return cells


def cell_from_concrete(X, eps):
    if any(np.isinf(c) for c in X):
        return None
    cell = np.floor(X / eps)

    # get the cell into integers...easier to do operations!

    cell_id = map(int, cell)
    return tuple(cell_id)


def ival_constraints(cell, eps):
    """ival_constraints

    Parameters
    ----------
    cell : cell id tuple
    eps : grid eps

    Returns
    -------

    Notes
    ------
    Cell is defined as left closed and right open: [ )
    This is achieved by subtracting a small % of eps from the cell's
    'right' boundary (ic.h)
    """
    tol = (1e-5)*np.array(eps) # 0.001% of eps
    cell_coordinates = np.array(cell) * eps
    ival_l = cell_coordinates
    ival_h = cell_coordinates + eps - tol
    return cons.IntervalCons(ival_l, ival_h)


def children_of(cell):
    """children_of
    Notes
    ------
    Gets the cells which will be contained in the passed in cell if
    the grid was refined s.t. new grid_eps = old grid_ps/2
    Equivalent to splitting the cell in every dimension.
    """
    #print(parent_cell_id)
    l = [[2*coord, 2*coord+1] for coord in cell]
    return list(itertools.product(*l))


def parent_of(cell):
    """parent of
    Notes
    ------
    Gets the cell which will be contain the passed in cell before the
    grid was refined, s.t. old grid_eps = 2*current grid_eps
    """
    # as i is an int, i/2 is equivalent to int(floor(i/2.0))
    # return tuple(int(floor(i/2.0)) for i in cell_id) # for clarity
    return tuple(i/2 for i in cell) # same as above


class Cell(object):

    def __init__(self, cell, eps):
        """__init__

        Parameters
        ----------
        abs_state : plant abstract state
        A : plant abstraction (for eps)
        """
        self.cell = cell
        self.eps = eps
        return

#     def split(self, axes=None):
#         '''verbose override of split_()'''
#         cells = self.split_(axes)
#         print '='*10
#         print 'splitting cell: {}'.format(self.ival_constraints)
#         for c in cells:
#             print c.ival_constraints
#         print '='*10
#         return cells

    def split(self, axes=None):
        """split

        Parameters
        ----------
        dims : dimensions to split along

        Returns
        -------
        child cells
        """
        if axes is None:
            return [Cell(i, self.eps/2) for i in get_children(self.cell)]
        else:
            cell = self.cell
            l = [[i] for i in cell]
            for i in axes:
                l[i] = [2*cell[i], 2*cell[i]+1]
            cells = list(itertools.product(*l))
            e_ = self.eps/2
            return [Cell(c, e_) for c in cells]

    @property
    def ival_constraints(self):
        return ival_constraints(self.cell, self.eps)

    def sample_UR(self, N):
        return self.ival_constraints.sample_UR(N)

    def __hash__(self):
        return hash((self.cell, tuple(self.eps)))

    def __eq__(self, c):
        return self.cell == c.cell and tuple(self.eps) == tuple(c.eps)

    def __str__(self):
        return str(self.cell)
