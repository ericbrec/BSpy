import numpy as np
import bspy.spline

def least_squares(dataPoints):
    rhsPoints = []
    uValues = []
    for dp in list(dataPoints):
        uValues.append(dp[0])
        rhsPoints.append(dp[1:])
