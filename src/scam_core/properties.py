class PropertyChecker():

    def __init__(self, final_cons):
        self.final_cons = final_cons
        return

    def check_array(self, T, Y):
        """check_array

        Parameters
        ----------
        T : Time Vector of the trace
        Y : Array of state vectors of the trace
        """
        # XXX: any() seems to work although np.any will be the right
        # function. Looks like any() just works on the array as an
        # iterator.
        return any(self.final_cons.contains(Y))
        #return np.any(self.final_cons.contains(Y))

    def first_sat_value_or_end(self, T, Y):
        """first_sat_value_or_end
        Gets the first satisfying state else returns the end of the
        trace.
        Parameters
        ----------
        T : Time Vector of the trace
        Y : Array of state vectors of the trace
        """
        sat = self.final_cons.contains(Y)
        if any(sat):
            return (T[sat][0], Y[sat][0]), True
        else:
            return (T[-1], Y[-1]), False

    def check(self, t, y):
        """check

        Parameters
        ----------
        t : time value
        y : state vector
        """
        return y in self.final_cons

    def __str__(self):
        return str(self.final_cons)


class PropertyCheckerNeverDetects():

    def __init__(self):
        return

    def check_array(self, T, Y):
        return False

    def check(self, t, y):
        return False
