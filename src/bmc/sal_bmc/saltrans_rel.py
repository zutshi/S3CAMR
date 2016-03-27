
'''
Creates a SAL transition system: Relational
'''

import itertools

import saltrans


# Make classes out of every header, prop, init, etc
class SALTransSysRel(saltrans.SALTransSys):

    def __init__(self, module_name, num_dim, init_cons, prop):
        super(SALTransSysRel, self).__init__(
                module_name, num_dim, init_cons, prop)
        # Store a mapping from a cell id: tuple -> sal loc name: str
        self.locs = {}
        self.id_ctr = itertools.count()
        return

    def get_loc_id(self, loc):
        """gets location id
        Assumes all locations have been added using add_locations.
        Will raise an error if the requested location has not been
        added."""
        #return self.locs.setdefault(loc, 'C' + str(next(self.id_ctr)))
        return self.locs[loc]

#     def add_loc(self, loc):
#         """adds location if not present already"""
#         if loc not in self.locs:
#             self.locs[loc] = 'C' + str(next(self.id_ctr))
#         return None

    def add_locations(self, locations):
        """add_locations
        Add all locations in one-shot

        Parameters
        ----------
        locations : locations/cells

        Notes
        ------
        Assumes every location in locations is unique
        """
        # just to make sure
        locations = set(locations)
        self.locs = {l: 'C' + str(next(self.id_ctr)) for l in locations}
        return

    @property
    def always_true_transition(self):
        return (super(SALTransSysRel, self).always_true_transition +
                ";\ncell' = cell")

    @property
    def type_decls(self):
        tp = super(SALTransSysRel, self).type_decls
        s = tp + '\nCELL: TYPE = {' + ', '.join(self.locs.itervalues()) + '};'
        return s

    @property
    def local_decls(self):
        ld = super(SALTransSysRel, self).local_decls
        return ld + '\nLOCAL\n\tcell:CELL'


Transition = saltrans.Transition
#     def __init__(self, name, g, r):
#         super(Transition, self).__init__(name, g, r)
#         #self.l1 = g.cell_id
#         #self.l2 = r.nex_cell_id
#         return


class Guard(saltrans.Guard):
    def __init__(self, cell_id, C, d):
        '''Cx <= d'''
        super(Guard, self).__init__(C, d)
        self.cell_id = cell_id

    def __str__(self):
        s = super(Guard, self).__str__()
        return 'cell = ' + self.cell_id + ' AND ' + s


class Reset(saltrans.Reset):
    def __init__(self, next_cell_id, A, b, error=None):
        super(Reset, self).__init__(A, b, error)
        self.next_cell_id = next_cell_id

    def __str__(self):
        s = super(Reset, self).__str__()
        return s + ";\ncell' = " + self.next_cell_id
