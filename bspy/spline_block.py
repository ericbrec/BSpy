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
    Spline blocks are useful for efficiently manipulating and solving systems of equations with splines.

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

    @staticmethod
    def _map_args(map, args):
        return [arg[map] if isinstance(arg, np.ndarray) else [arg[index] for index in map] for arg in args]
    
    def _block_evaluation(self, returnShape, splineFunction, args):
        value = np.zeros(returnShape, self.coefsDtype)
        nDep = 0
        for row in self.block:
            for map, spline in row:
                value[nDep:nDep + spline.nDep] += splineFunction(spline, *SplineBlock._map_args(map, args))
            nDep += spline.nDep
        return value
    
    def _block_operation(self, splineFunction, args):
        newBlock = []
        for row in self.block:
            newRow = []
            for map, spline in row:
                newRow.append((map, splineFunction(spline, *SplineBlock._map_args(map, args))))
            newBlock.append(newRow)
        return SplineBlock(newBlock)
    
    def __init__(self, block):
        if isinstance(block, bspy.spline.Spline):
            block = [[block]]
        elif isinstance(block[0], bspy.spline.Spline) or (len(block) > 1 and isinstance(block[1], bspy.spline.Spline)):
            block = [block]

        self.block = []
        self.nInd = 0
        self.nDep = 0
        self.knotsDtype = None
        self.coefsDtype = None
        self.size = 0
        domain = {}
        for row in block:
            rowInd = 0
            rowDep = 0
            indSet = set()
            newRow = []
            for entry in row:
                if isinstance(entry, bspy.spline.Spline):
                    spline = entry
                    map = list(range(rowInd, rowInd + spline.nInd))
                else:
                    (map, spline) = entry
                rowInd += spline.nInd
                if rowDep == 0:
                    rowDep = spline.nDep
                    if self.nDep == 0:
                        self.knotsDtype = spline.knots[0].dtype
                        self.coefsDtype = spline.coefs.dtype
                elif rowDep != spline.nDep:
                    raise ValueError("All splines in the same row must have the same nDep")
                d = spline.domain()
                for ind, i in enumerate(map):
                    if i in indSet:
                        raise ValueError(f"Multiple splines in the same row map to independent variable {i}")
                    else:
                        indSet.add(i)
                    if i in domain:
                        if domain[i][0] != d[ind, 0] or domain[i][1] != d[ind, 1]:
                            raise ValueError("Domains of independent variables must match")
                    else:
                        domain[i] = d[ind]
                newRow.append((map, spline))
            
            if rowDep > 0:
                self.nDep += rowDep
                self.size += len(row)
                self.block.append(newRow)

        self.nInd = len(domain)
        self._domain = []
        for i in range(self.nInd):
            if i in domain:
                self._domain.append(domain[i])
            else:
                raise ValueError(f"Block is missing independent variable {i}")
        self._domain = np.array(self._domain, self.knotsDtype)

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
        `zeros` : Find the roots of a block of splines (nInd must match nDep).

        Notes
        -----
        Uses `zeros` to find all intersection points and `Spline.contour` to find individual intersection curves. 
        The algorithm used to to find all intersection curves is from Grandine, Thomas A., and Frederick W. Klein IV. 
        "A new approach to the surface intersection problem." Computer Aided Geometric Design 14, no. 2 (1997): 111-134.
        """
        return bspy._spline_intersection.contours(self)

    def contract(self, uvw):
        """
        Contract a spline block by assigning a fixed value to one or more of its independent variables.

        Parameters
        ----------
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable to contract.
            A value of `None` for an independent variable indicates that variable should remain unchanged.

        Returns
        -------
        block : `SplineBlock`
            The contracted spline block.
        """
        # First, remap the independent variables.
        remap = []
        newIndex = 0
        for value in uvw:
            if value is not None:
                remap.append(newIndex)
                newIndex += 1
            else:
                remap.append(None)

        # Next, rebuild the block with contracted splines.
        newBlock = []
        for row in self.block:
            newRow = []
            for map, spline in row:
                spline = spline.contract([uvw[index] for index in map])
                map = [remap[ind] for ind in map if uvw[ind] is not None]
                newRow.append((map, spline))
            newBlock.append(newRow)
        return SplineBlock(newBlock)

    def derivative(self, with_respect_to, uvw):
        """
        Compute the derivative of the spline block at given parameter values.

        Parameters
        ----------
        with_respect_to : `iterable`
            An iterable of length `nInd` that specifies the integer order of derivative for each independent variable.
            A zero-order derivative just evaluates the spline normally.
        
        uvw : `iterable`
            An iterable of length `nInd` that specifies the values of each independent variable (the parameter values).

        Returns
        -------
        value : `numpy.array`
            The value of the derivative of the spline block at the given parameter values (array of size nDep).
        """
        return self._block_evaluation(self.nDep, bspy._spline_evaluation.derivative, (with_respect_to, uvw))

    def domain(self):
        """
        Return the domain of a spline block.

        Returns
        -------
        bounds : `numpy.array`
            nInd x 2 array of the lower and upper bounds on each of the independent variables.
        """
        return self._domain
    
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
        return self._block_evaluation(self.nDep, bspy._spline_evaluation.evaluate, (uvw,))
    
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
        jacobian = np.zeros((self.nDep, self.nInd), self.coefsDtype)
        uvw = np.atleast_1d(uvw)
        nDep = 0
        for row in self.block:
            for map, spline in row:
                jacobian[nDep:nDep + spline.nDep, map] += spline.jacobian(uvw[map])
            nDep += spline.nDep
        return jacobian

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

        See Also
        --------
        `normal_spline` : Compute a spline that evaluates to the normal of the given spline block (not normalized).

        Notes
        -----
        Attentive readers will notice that the number of independent variables could be one more than the number of 
        dependent variables (instead of one less, as is typical). In that case, the normal represents the null space of 
        the matrix formed by the tangents of the spline. If the null space is greater than one dimension, the normal will be zero.
        """
        return bspy._spline_evaluation.normal(self, uvw, normalize, indices)

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
        `normal` : Compute the normal of the spline block at given parameter values.

        Notes
        -----
        Attentive readers will notice that the number of independent variables could be one more than the number of 
        dependent variables (instead of one less, as is typical). In that case, the normal represents the null space of 
        the matrix formed by the tangents of the spline block. If the null space is greater than one dimension, the normal will be zero.
        """
        return bspy._spline_operations.normal_spline(self, indices)
    
    def range_bounds(self):
        """
        Return the range of a spline block as lower and upper bounds on each of the
        dependent variables.
        """
        return self._block_evaluation((self.nDep, 2), bspy._spline_evaluation.range_bounds, [])
    
    def reparametrize(self, newDomain):
        """
        Reparametrize a spline block to match new domain bounds. The number of knots and coefficients remain unchanged.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new lower and upper bounds on each of the independent variables (same form as 
            returned from `domain`). If a bound pair is `None` then the original bound (and knots) are left unchanged. 
            For example, `[[0.0, 1.0], None]` will reparametrize the first independent variable and leave the second unchanged)

        Returns
        -------
        block : `SplineBlock`
            Reparametrized spline block.

        See Also
        --------
        `domain` : Return the domain of a spline block.
        """
        return self._block_operation(bspy._spline_domain.reparametrize, (newDomain,))

    def trim(self, newDomain):
        """
        Trim the domain of a spline block.

        Parameters
        ----------
        newDomain : array-like
            nInd x 2 array of the new lower and upper bounds on each of the independent variables (same form as 
            returned from `domain`). If a bound is None or nan then the original bound (and knots) are left unchanged.

        Returns
        -------
        block : `SplineBlock`
            Trimmed spline block.
        """
        return self._block_operation(bspy._spline_domain.trim, (newDomain,))
    
    def zeros(self, epsilon=None, initialScale=None):
        """
        Find the roots of a block of splines (nInd must match nDep).

        Parameters
        ----------
        epsilon : `float`, optional
            Tolerance for root precision. The root will be within epsilon of the actual root. 
            The default is the machine epsilon.

        initialScale : array-like, optional
            The initial scale of each dependent variable (as opposed to the current scale of 
            the spline block, which may have been normalized). The default is an array of ones (size nDep).

        Returns
        -------
        roots : `iterable`
            An iterable containing the roots of the block of splines. If the block is 
            zero over an interval, that root will appear as a tuple of the interval. 
            For curves (nInd == 1), the roots are ordered.
        
        See Also
        --------
        `contours` : Find all the contour curves of a spline block whose `nInd` is one larger than its `nDep`.

        Notes
        -----
        Implements a variation of the projected-polyhedron technique from Sherbrooke, Evan C., and Nicholas M. Patrikalakis. 
        "Computation of the solutions of nonlinear polynomial systems." Computer Aided Geometric Design 10, no. 5 (1993): 379-405.
        """
        return bspy._spline_intersection.zeros_using_projected_polyhedron(self, epsilon, initialScale)
