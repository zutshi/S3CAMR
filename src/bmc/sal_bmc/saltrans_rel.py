
'''
Creates a SAL transition system: Relational
'''

import itertools
import collections

import saltrans


# Make classes out of every header, prop, init, etc
class SALTransSysRel(saltrans.SALTransSys):

    def __init__(self, module_name, vs, init_cons, prop):
        super(SALTransSysRel, self).__init__(
                module_name, vs, init_cons, prop)
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

    def add_location(self, location):
        """Add a location

        Parameters
        ----------
        locations : location/cell id

        Notes
        ------
        Assumes every location is hashable and unique, else an
        overwrite will occur!
        """
        # TODO: Fix the ugliness using setdefault
        if location in self.locs:
            return self.locs[location]
        else:
            self.locs[location] = 'C' + str(next(self.id_ctr))
            return self.locs[location]
        return

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
    def type_decls(self):
        tp = super(SALTransSysRel, self).type_decls
        s = tp + '\nCELL: TYPE = {' + ', '.join(self.locs.itervalues()) + '};'
        return s

    @property
    def local_decls(self):
        ld = super(SALTransSysRel, self).local_decls
        return ld + '\nLOCAL\n\tcell:CELL'


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
