class ArgumentOutsideDomainError(Exception):
    """Exception raised for evaluating a spline outside its domain """
    def __init__(self, uvw,
                 message = "Spline evaluation outside domain"):
        self.uvw = uvw
        self.message = message
        super().__init__(self.message)

class InvalidCoefficientError(Exception):
    """Exception raised for invalid B-spline coefficient"""
    def __init__(self, coefs,
                 message = "Invalid B-spline coefficient"):
        self.coefs = coefs
        self.message = message
        super().__init__(self.message)

class InvalidKnotsError(Exception):
    """Exception raised for invalid number of knot arrays"""
    def __init__(self, nknots,
                 message = "Invalid number of knot arrays"):
        self.nknots = nknots
        self.message = message
        super().__init__(self.message)

class InvalidKnotsError(Exception):
    """Exception raised for invalid number of knot arrays"""
    def __init__(self, nknots,
                 message = "Invalid number of knot arrays"):
        self.nknots = nknots
        self.message = message
        super().__init__(self.message)

class InvalidNdepError(Exception):
    """Exception raised for invalid number of dependent variables"""
    def __init__(self, ndep,
                 message = "Ndep is not a non-negative integer"):
        self.ndep = ndep
        self.message = message
        super().__init__(self.message)

class InvalidNindError(Exception):
    """Exception raised for invalid number of independent variables"""
    def __init__(self, nind,
                 message = "Nind is not a non-negative integer"):
        self.nind = nind
        self.message = message
        super().__init__(self.message)

class InvalidNcoefsError(Exception):
    """Exception raised for invalid number of B-spline coefficients"""
    def __init__(self, ncoefs,
                 message = "Invalid number of B-spline coefficients"):
        self.ncoefs = ncoefs
        self.message = message
        super().__init__(self.message)

class InvalidNknotsError(Exception):
    """Exception raised for invalid number of knot arrays"""
    def __init__(self, nknots,
                 message = "Should be Nind knot arrays"):
        self.nknots = nknots
        self.message = message
        super().__init__(self.message)
        
class InvalidNorderError(Exception):
    """Exception raised for invalid number of polynomial orders"""
    def __init__(self, order,
                 message = "Should be Nind polynomial orders"):
        self.order = order
        self.message = message
        super().__init__(self.message)


