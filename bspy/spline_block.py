import numpy as np
import bspy.spline
import bspy._spline_domain
import bspy._spline_evaluation
import bspy._spline_intersection
import bspy._spline_fitting
import bspy._spline_operations

class SplineBlock:
    """
    A class to represent and process an array-like collection of splines.

    Parameters
    ----------
    block : an array-like collection of splines (may be a list of lists)
        Splines in the same row are treated as if they are added together. 
        Each row need not have the same number of splines, but all splines in a row must have the same 
        number of dependent variables (same nDep). Corresponding independent variables must have the same domain.

        For example, if F is a spline with nInd = 3 and nDep = 2, G is a spline with nInd = 1 and nDep = 2, 
        and h is a spline with nInd = 2 and nDep = 1, then [[F, G], [h]] is a valid block (total nDep is 3 
        and max nInd is 4).
    """
    
    def __init__(self, block):
        if isinstance(block, bspy.spline.Spline):
            self.block = [[block]]
        elif isinstance(block[0], bspy.spline.Spline):
            self.block = [block]
        else:
            self.block = block
        self.nInd = 0
        self.nDep = 0
        self.knotsDtype = None
        self.coefsDtype = None
        self.domain = []
        for row in self.block:
            rowInd = 0
            rowDep = 0
            for spline in row:
                if rowDep == 0:
                    rowDep = spline.nDep
                    if self.nInd == 0:
                        self.knotsDtype = spline.knots[0].dtype
                        self.coefsDtype = spline.coefs.dtype
                elif rowDep != spline.nDep:
                    raise ValueError("All splines in the same row must have the same nDep")
                d = spline.domain()
                for i in range(rowInd, rowInd + spline.nInd):
                    ind = i - rowInd
                    if i < self.nInd:
                        if self.domain[i][0] != d[ind, 0] or self.domain[i][1] != d[ind, 1]:
                            raise ValueError("Domains of independent variables must match")
                    else:
                        self.domain.append(d[ind])
                rowInd += spline.nInd
            self.nInd = max(self.nInd, rowInd)
            self.nDep += rowDep
        self.domain = np.array(self.domain, self.knotsDtype)

    def __call__(self, uvw):
        return self.evaluate(uvw)

    def __repr__(self):
        return f"SplineBlock({self.block})"

    def contours(self):
        """
        Find all the contour curves of a block of splines whose `nInd` is one larger than its `nDep`.

        Returns
        -------
        curves : `iterable`
            A collection of `Spline` curves, `u(t)`, each of whose domain is [0, 1], whose range is
            in the parameter space of the given spline, and which satisfy `self(u(t)) = 0`. 

        See Also
        --------
        `contours` : Find all the contour curves of a spline whose `nInd` is one larger than its `nDep`.
        `contour` : Fit a spline to the contour defined by `F(x) = 0`.
        `zeros` : Find the roots of a spline (nInd must match nDep).
        `zeros_extended` : Find the roots of a block of splines (nInd must match nDep).
        `intersect` : Intersect two splines.

        Notes
        -----
        Uses `zeros` to find all intersection points and `contour` to find individual intersection curves. 
        The algorithm used to to find all intersection curves is from Grandine, Thomas A., and Frederick W. Klein IV. 
        "A new approach to the surface intersection problem." Computer Aided Geometric Design 14, no. 2 (1997): 111-134.
        """
        return bspy._spline_intersection.contours(self)
    
    def evaluate(self, uvw):
        """
        Compute the value of the spline block at given parameter values.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).

        Returns
        -------
        value : `numpy.array`
            The value of the spline block at the given parameter values (array of size nDep).
        """
        return bspy._spline_evaluation.block_evaluate(self, uvw)
    
    def jacobian(self, uvw):
        """
        Compute the value of the spline block's Jacobian at given parameter values.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).

        Returns
        -------
        value : `numpy.array`
            The value of the spline block's Jacobian at the given parameter values. The shape of the return value is (nDep, nInd).
        """
        return bspy._spline_evaluation.block_jacobian(self, uvw)

    def normal(self, uvw, normalize=True, indices=None):
        """
        Compute the normal of the spline block at given parameter values. The number of independent variables must be
        one different than the number of dependent variables.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).
        
        normalize : `boolean`, optional
            If True the returned normal will have unit length (the default). Otherwise, the normal's length will
            be the area of the tangent space (for two independent variables, its the length of the cross product of tangent vectors).
        
        indices : `iterable`, optional
            An iterable of normal indices to calculate. For example, `indices=(0, 3)` will return a vector of length 2
            with the first and fourth values of the normal. If `None`, all normal values are returned (the default).

        Returns
        -------
        normal : `numpy.array`
            The normal vector of the spline block at the given parameter values.

        Notes
        -----
        Attentive readers will notice that the number of independent variables could be one more than the number of 
        dependent variables (instead of one less, as is typical). In that case, the normal represents the null space of 
        the matrix formed by the tangents of the spline. If the null space is greater than one dimension, the normal will be zero.
        """
        return bspy._spline_evaluation.block_normal(self, uvw, normalize, indices)

    def normal_spline(self, indices=None):
        """
        Compute a spline that evaluates to the normal of the given spline block. The length of the normal
        is the area of the tangent space (for two independent variables, its the length of the cross product of tangent vectors).
        The number of independent variables must be one different than the number of dependent variables.
        Find all the contour curves of a block of splines whose `nInd` is one larger than its `nDep`.

        Parameters
        ----------
        indices : `iterable`, optional
            An iterable of normal indices to calculate. For example, `indices=(0, 3)` will make the returned spline compute a vector of length 2
            with the first and fourth values of the normal. If `None`, all normal values are returned (the default).

        Returns
        -------
        spline : `Spline`
            The spline that evaluates to the normal of the given spline block.

        See Also
        --------
        `normal` : Compute the normal of the spline at given parameter values.
        `normal_spline` : Compute a spline that evaluates to the normal of the given spline (not normalized).

        Notes
        -----
        Attentive readers will notice that the number of independent variables could be one more than the number of 
        dependent variables (instead of one less, as is typical). In that case, the normal represents the null space of 
        the matrix formed by the tangents of the spline block. If the null space is greater than one dimension, the normal will be zero.
        """
        return bspy._spline_operations.normal_spline(self, indices)
    
    def zeros(self, epsilon=None):
        """
        Find the roots of a block of splines (nInd must match nDep).

        Parameters
        ----------
        epsilon : `float`, optional
            Tolerance for root precision. The root will be within epsilon of the actual root. 
            The default is the machine epsilon.

        Returns
        -------
        roots : `iterable`
            An iterable containing the roots of the block of splines. If the block is 
            zero over an interval, that root will appear as a tuple of the interval. 
            For curves (nInd == 1), the roots are ordered.
        
        See Also
        --------
        `zeros` : Find the roots of a spline.
        `intersect` : Intersect two splines.
        `contours` : Find all the contour curves of a spline whose `nInd` is one larger than its `nDep`.
        `contour_extended` : Find all the contour curves of a block of splines whose `nInd` is one larger than its `nDep`.

        Notes
        -----
        Implements a variation of the projected-polyhedron technique from Sherbrooke, Evan C., and Nicholas M. Patrikalakis. 
        "Computation of the solutions of nonlinear polynomial systems." Computer Aided Geometric Design 10, no. 5 (1993): 379-405.
        """
        return bspy._spline_intersection.zeros_using_projected_polyhedron(self, epsilon)
