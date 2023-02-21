import numpy as np
from bspy import Spline

class Polynomial:
    """
    A class to represent, convert, and process 1-D polynomials: sum i=0..degree of ci * (x - x0) ^ i.
    This class acts as a helper class for the spline class, providing
    operations that are difficult to compute otherwise, such as multiplication.

    Parameters
    ----------
    order : `int`
        The order of polynomial (one higher than the degree)
    
    c : array-like
        The polynomial coefficients

    x0 : `float`, optional
        The center point of the polynomial (default is 0)
    """
    def __init__(self, order, c, x0 = 0):
        assert order > 0, "order <= 0"
        assert len(c) == order, "len(a) != order"

        self.order = order
        self.c = np.array(c)
        self.x0 = x0

    def __repr__(self):
        return f"Polynomial({self.order}, {self.c}, {self.x0})"

    def evaluate(self, x):
        """
        Compute the value of a polynomial at given parameter value.

        Parameters
        ----------
        x : `float`
            The value of the parameter
        
        Returns
        -------
        value : `numpy.array`
            The value of the polynomial, sum i=0..degree of ci * (x - x0) ^ i
        """
        delta = x - self.x0
        degree = self.order - 1
        value = self.c[degree]
        for i in range(1, self.order):
            value = self.c[degree-i] + value * delta
        return value

    def multiply(self, other):
        """
        Multiply two polynomials the value of a polynomial at given parameter value.

        Parameters
        ----------
        other : `Polynomial`
            The polynomial to multiple by self. Note that other gets shifted to the same center as self.
        
        Returns
        -------
        polynomial : `Polynomial`
            The result of multiplying other by self

        Notes
        -----
        Taken from Lee, E. T. Y. "Computing a chain of blossoms, with application to products of splines." 
        Computer Aided Geometric Design 11, no. 6 (1994): 597-620.
        """
        order = self.order + other.order - 1
        shape = [*self.c.shape]
        shape[0] = order
        c = np.zeros(shape, self.c.dtype)
        other.shift(self.x0) # Force both polynomials to have the same center.
        for j in range(other.order):
            for i in range(j, self.order + j):
                c[i] += self.c[i - j] * other.c[j]
        return Polynomial(order, c, self.x0)

    def raceme(self, m, rho, v):
        """
        Compute the raceme blossom chain for a polynomial at given parameter values.

        Parameters
        ----------
        m : `int`
            The width of the blossom
        
        rho : `int`
            The number of blossoms to compute
        
        v : array-like
            The parameter values for the blossoms

        Returns
        -------
        a : `numpy.array`
            The values for the `rho` number of blossoms at the given parameters

        Notes
        -----
        Taken from Lee, E. T. Y. "Computing a chain of blossoms, with application to products of splines." 
        Computer Aided Geometric Design 11, no. 6 (1994): 597-620.
        """
        assert self.order - 1 <= m, "degree > m"
        assert rho > 0, "rho <= 0"
        assert len(v) >= m, "len(v) < m"

        shape = [*self.c.shape]
        shape[0] = max(rho, self.order)
        a = np.zeros(shape, self.c.dtype)
        a[:self.order] = self.c
        for j in range(m):
            for i in range(min(self.order, m - j)):
                a[i] = (1 - i/(m - j)) * a[i] + ((i + 1)/(m - j)) * (v[m - j - 1] - self.x0) * a[i + 1]
        for j in range(rho - 1):
            for i in range(min(self.order + j, rho - 1), j, -1):
                a[i] = a[i - 1] + (v[m+j] - v[i - 1]) * a[i]
        return a[:rho]

    def shift(self, x0 = 0):
        """
        Shift the center point of the polynomial to a new x0.

        Parameters
        ----------
        x0 : `float`, optional
            The new center point of the polynomial (default is 0)

        Notes
        -----
        There is no return value, the polynomial is changed in place.
        Taken from Lee, E. T. Y. "Computing a chain of blossoms, with application to products of splines." 
        Computer Aided Geometric Design 11, no. 6 (1994): 597-620.
        """
        delta = x0 - self.x0
        if abs(delta) > np.finfo(float).eps:
            for j in range(self.order):
                for i in range(self.order - 1, j, -1):
                    self.c[i - 1] += delta * self.c[i]
            self.x0 = x0