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
        return self.final_cons.contains(Y)

    def check(self, t, y):
        """check

        Parameters
        ----------
        t : time value
        y : state vector
        """
        return y in self.final_cons
