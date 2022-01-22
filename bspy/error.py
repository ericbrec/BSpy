class ArgumentOutsideDomainError(Exception):
    """Exception raised for evaluating a spline outside its domain """
    def __init__(self, uvw,
                 message = "Spline evaluation outside domain"):
        self.uvw = uvw
        self.message = message
        super().__init__(self.message)

