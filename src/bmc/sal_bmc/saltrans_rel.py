from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


'''
Creates a SAL transition system: Relational
'''

import itertools
import collections

from . import saltrans


# Make classes out of every header, prop, init, etc
class SALTransSysRel(saltrans.SALTransSys):

    def __init__(self, module_name, vs, init_cons, prop):
        super(SALTransSysRel, self).__init__(module_name, vs, init_cons, prop)
        # Store a mapping from a cell id: tuple -> sal loc name: str
        self.partid2Cid = {}
        self.id_ctr = itertools.count()
        return

    def get_C(self, partid):
        """Gets the bmc Cid corresponding to a pwa partition id
        Assumes all Cid have been added using add_C.
        Will raise an error if the requested location has not been
        added."""
        #return self.partid2Cid.setdefault(loc, 'C' + str(next(self.id_ctr)))
        return self.partid2Cid[partid]

#     def add_loc(self, loc):
#         """adds location if not present already"""
#         if loc not in self.partid2Cid:
#             self.partid2Cid[loc] = 'C' + str(next(self.id_ctr))
#         return None

    def add_C(self, c):
        """Add a cell

        Parameters
        ----------
        c : cell id

        Notes
        ------
        Assumes every C is hashable and unique, else an
        overwrite will occur!
        """
        # TODO: Fix the ugliness using setdefault
#         if c in self.partid2Cid:
#             return self.partid2Cid[c]
#         else:
#             self.partid2Cid[c] = 'C' + str(next(self.id_ctr))
#             return self.partid2Cid[c]
#         return
        # simplified equivalent code from above
        if c not in self.partid2Cid:
            self.partid2Cid[c] = 'C' + str(next(self.id_ctr))
        return self.partid2Cid[c]

    def add_Cs(self, Cs):
        """add_Cs
        Add all locations/cells in one-shot

        Parameters
        ----------
        Cs : cells

        Notes
        ------
        Assumes every location in locations is unique
        """
        # just to make sure
        Cs = set(Cs)
        self.partid2Cid = {c: 'C' + str(next(self.id_ctr)) for c in Cs}
        return

    @property
    def type_decls(self):
        tp = super(SALTransSysRel, self).type_decls
        s = tp + '\nCELL: TYPE = {' + ', '.join(self.partid2Cid.values()) + '};'
        return s

    @property
    def local_decls(self):
        return ''
        #ld = super(SALTransSysRel, self).local_decls
        #return ld + '\nLOCAL\n\tcell:CELL'

    @property
    def op_decls(self):
        op = super(SALTransSysRel, self).op_decls
        return op + ',\n\tcell:CELL'

Transition = saltrans.Transition


class Guard(saltrans.Guard):
    def __init__(self, cell_id, C, d):
        '''Cx <= d'''
        super(Guard, self).__init__(C, d)
        self.cell_id = cell_id

    def __str__(self):
        s = super(Guard, self).__str__()
        return 'cell = ' + self.cell_id + ' AND ' + s


class Reset(saltrans.Reset):
    def __init__(self, next_cell_ids, A, b, error):
        super(Reset, self).__init__(A, b, error)
        self.next_cell_ids = next_cell_ids

    def __str__(self):
        s = super(Reset, self).__str__()
        assert(isinstance(self.next_cell_ids, collections.Iterable))
        return s + ";\ncell' IN {" + ', '.join(self.next_cell_ids) + "}"
