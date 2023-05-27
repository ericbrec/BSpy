import numpy as np
from collections import namedtuple

def zeros(self, epsilon=None):
    assert self.nInd == self.nDep
    assert self.nInd == 1
    machineEpsilon = np.finfo(self.knots[0].dtype).eps
    if epsilon is None:
        epsilon = max(self.accuracy, machineEpsilon)
    roots = []
    # Set initial spline, domain, and interval.
    spline = self
    (domain,) = spline.domain()
    Interval = namedtuple('Interval', ('spline', 'slope', 'intercept', 'atMachineEpsilon'))
    intervalStack = [Interval(spline.trim((domain,)).reparametrize(((0.0, 1.0),)), domain[1] - domain[0], domain[0], False)]

    def test_and_add_domain():
        """Macro to perform common operations when considering a domain as a new interval."""
        if domain[0] <= 1.0 and domain[1] >= 0.0:
            width = domain[1] - domain[0]
            if width >= 0.0:
                slope = width * interval.slope
                intercept = domain[0] * interval.slope + interval.intercept
                # Iteration is complete if the interval actual width (slope) is either
                # one iteration past being less than sqrt(machineEpsilon) or simply less than epsilon.
                if interval.atMachineEpsilon or slope < epsilon:
                    root = intercept + 0.5 * slope
                    # Double-check that we're at an actual zero (avoids boundary case).
                    if self((root,)) < epsilon:
                        # Check for duplicate root. We test for a distance between roots of 2*epsilon to account for a left vs. right sided limit.
                        if roots and abs(root - roots[-1]) < 2.0 * epsilon:
                            # For a duplicate root, return the average value.
                            roots[-1] = 0.5 * (roots[-1] + root)
                        else:
                            roots.append(root)
                else:
                    intervalStack.append(Interval(spline.trim((domain,)).reparametrize(((0.0, 1.0),)), slope, intercept, slope * slope < machineEpsilon))

    # Process intervals until none remain
    while intervalStack:
        interval = intervalStack.pop()
        range = interval.spline.range_bounds()
        scale = np.abs(range).max(axis=1)
        if scale < epsilon:
            roots.append((interval.intercept, interval.slope + interval.intercept))
        else:
            spline = interval.spline.scale(1.0 / scale)
            mValue = spline((0.5,))
            derivativeRange = spline.differentiate().range_bounds()
            if derivativeRange[0, 0] * derivativeRange[0, 1] <= 0.0:
                # Derivative range contains zero, so consider two intervals.
                leftIndex = 0 if mValue > 0.0 else 1
                domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                domain[1] = 1.0
                test_and_add_domain()
                domain[0] = 0.0
                domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                test_and_add_domain()
            else:
                leftIndex = 0 if mValue > 0.0 else 1
                domain[0] = max(0.5 - mValue / derivativeRange[0, leftIndex], 0.0)
                domain[1] = min(0.5 - mValue / derivativeRange[0, 1 - leftIndex], 1.0)
                test_and_add_domain()
    
    return roots