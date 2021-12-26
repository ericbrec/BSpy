import numpy as np
import quaternion as quat
import tkinter as tk
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from pyopengltk import OpenGLFrame
from bspy.spline import Spline

class SplineOpenGLFrame(OpenGLFrame):

    computeBasisCode = """
        void ComputeBasis(in int offset, in int order, in int n, in int m, in float u, 
            out float uBasis[{maxBasis}], out float duBasis[{maxBasis}])
        {{
            int degree = 1;

            uBasis = float[{maxBasis}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            duBasis = float[{maxBasis}](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            uBasis[order-1] = 1.0;

            while (degree < order - 1)
            {{
                int b = order - degree - 1;
                for (int i = m - degree; i < m + 1; i++)
                {{
                    float gap0 = texelFetch(uSplineData, offset + i + degree).x - texelFetch(uSplineData, offset + i).x; // knots[i+degree] - knots[i]
                    float gap1 = texelFetch(uSplineData, offset + i + degree + 1).x - texelFetch(uSplineData, offset + i + 1).x; // knots[i+degree+1] - knots[i+1]
                    gap0 = gap0 < 1.0e-8 ? 0.0 : 1.0 / gap0;
                    gap1 = gap1 < 1.0e-8 ? 0.0 : 1.0 / gap1;

                    float val0 = (u - texelFetch(uSplineData, offset + i).x) * gap0; // (u - knots[i]) * gap0;
                    float val1 = (texelFetch(uSplineData, offset + i + degree + 1).x - u) * gap1; // (knots[i+degree+1] - u) * gap1;
                    uBasis[b] = uBasis[b] * val0 + uBasis[b+1] * val1;
                    b++;
                }}
                degree++;
            }}
            while (degree < order)
            {{
                int b = order - degree - 1;
                for (int i = m - degree; i < m + 1; i++)
                {{
                    float gap0 = texelFetch(uSplineData, offset + i + degree).x - texelFetch(uSplineData, offset + i).x; // knots[i+degree] - knots[i]
                    float gap1 = texelFetch(uSplineData, offset + i + degree + 1).x - texelFetch(uSplineData, offset + i + 1).x; // knots[i+degree+1] - knots[i+1]
                    gap0 = gap0 < 1.0e-8 ? 0.0 : 1.0 / gap0;
                    gap1 = gap1 < 1.0e-8 ? 0.0 : 1.0 / gap1;

                    float d0 = degree * gap0;
                    float d1 = -degree * gap1;
                    duBasis[b] = uBasis[b] * d0 + uBasis[b+1] * d1;
                    float val0 = (u - texelFetch(uSplineData, offset + i).x) * gap0; // (u - knots[i]) * gap0;
                    float val1 = (texelFetch(uSplineData, offset + i + degree + 1).x - u) * gap1; // (knots[i+degree+1] - u) * gap1;
                    uBasis[b] = uBasis[b] * val0 + uBasis[b+1] * val1;
                    b++;
                }}
                degree++;
            }}
        }}
    """

    computeDeltaCode = """
        void ComputeDelta(in vec4 point, in vec3 dPoint, in vec3 d2Point, inout float delta)
        {
            float zScale = 1.0 / (uScreenScale.z - point.z);
            float zScale2 = zScale * zScale;
            float zScale3 = zScale2 * zScale;
            vec2 projection = uScreenScale.z > 1.0 ? 
                vec2(uScreenScale.x * (d2Point.x * zScale - 2.0 * dPoint.x * dPoint.z * zScale2 +
                    point.x * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2)),
                    uScreenScale.y * (d2Point.y * zScale - 2.0 * dPoint.y * dPoint.z * zScale2 +
                    point.y * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2)))
                : vec2(uScreenScale.x * d2Point.x, uScreenScale.y * d2Point.y);
            float projectionLength = length(projection);
            float newDelta = projectionLength < 1.0e-8 ? delta : 1.0 / sqrt(projectionLength);
            delta = min(newDelta, delta);
        }
    """

    curveVertexShaderCode = """
        #version 410 core
     
        const int header = 2;

        attribute vec4 aParameters;

        uniform samplerBuffer uSplineData;

        out SplineInfo
        {
            int uOrder;
            int uN;
            int uM;
            float firstU;
            float lastU;
            float deltaU;
        } outData;

        void main()
        {
            outData.uOrder = int(texelFetch(uSplineData, 0).x);
            outData.uN = int(texelFetch(uSplineData, 1).x);
            outData.uM = min(gl_InstanceID + outData.uOrder - 1, outData.uN - 1);
            outData.firstU = texelFetch(uSplineData, header + outData.uM).x; // knots[uM]
            outData.lastU = texelFetch(uSplineData, header + outData.uM + 1).x; // knots[uM+1]
            outData.deltaU = outData.lastU - outData.firstU;
            gl_Position = aParameters;
        }
    """

    computeCurveSamplesCode = """
        void ComputeCurveSamples(out float uSamples)
        {
            float uInterval = outData.deltaU; // uKnots[uM+1] - uKnots[uM]

            outData.deltaU = max(uInterval, 1.0e-8);
            int i = outData.uM+1-outData.uOrder;
            int coefficientOffset = header + outData.uOrder + outData.uN + 4 * i;
            vec4 coefficient0 = vec4(
                texelFetch(uSplineData, coefficientOffset+0).x, 
                texelFetch(uSplineData, coefficientOffset+1).x,
                texelFetch(uSplineData, coefficientOffset+2).x,
                texelFetch(uSplineData, coefficientOffset+3).x);
            coefficientOffset += 4;
            vec4 coefficient1 = vec4(
                texelFetch(uSplineData, coefficientOffset+0).x, 
                texelFetch(uSplineData, coefficientOffset+1).x,
                texelFetch(uSplineData, coefficientOffset+2).x,
                texelFetch(uSplineData, coefficientOffset+3).x);
            float gap = texelFetch(uSplineData, header + i+outData.uOrder).x - texelFetch(uSplineData, header + i+1).x; // uKnots[i+uOrder] - uKnots[i+1]
            gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
            vec3 dPoint0 = (outData.uOrder - 1) * gap * (coefficient1.xyz - coefficient0.xyz);
            while (i < outData.uM-1)
            {
                coefficientOffset += 4;
                vec4 coefficient2 = vec4(
                    texelFetch(uSplineData, coefficientOffset+0).x, 
                    texelFetch(uSplineData, coefficientOffset+1).x,
                    texelFetch(uSplineData, coefficientOffset+2).x,
                    texelFetch(uSplineData, coefficientOffset+3).x);
                gap = texelFetch(uSplineData, header + i+1+outData.uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+1+uOrder] - uKnots[i+2]
                gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                vec3 dPoint1 = (outData.uOrder - 1) * gap * (coefficient2.xyz - coefficient1.xyz);
                gap = texelFetch(uSplineData, header + i+outData.uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+uOrder] - uKnots[i+2]
                gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                vec3 d2Point = (outData.uOrder - 2) * gap * (dPoint1 - dPoint0);

                ComputeDelta(coefficient0, dPoint0, d2Point, outData.deltaU);
                ComputeDelta(coefficient1, dPoint0, d2Point, outData.deltaU);
                ComputeDelta(coefficient1, dPoint1, d2Point, outData.deltaU);
                ComputeDelta(coefficient2, dPoint1, d2Point, outData.deltaU);

                coefficient0 = coefficient1;
                coefficient1 = coefficient2;
                dPoint0 = dPoint1;
                i++;
            }
            uSamples = ceil(uInterval / outData.deltaU);
            outData.deltaU = uInterval / uSamples;
        }
    """

    curveTCShaderCode = """
        #version 410 core

        layout (vertices = 1) out;

        const int header = 2;

        in SplineInfo
        {{
            int uOrder;
            int uN;
            int uM;
            float firstU;
            float lastU;
            float deltaU;
        }} inData[];

        uniform vec3 uScreenScale;
        uniform samplerBuffer uSplineData;

        patch out SplineInfo
        {{
            int uOrder;
            int uN;
            int uM;
            float firstU;
            float lastU;
            float deltaU;
        }} outData;

        {computeDeltaCode}

        {computeCurveSamplesCode}

        void main()
        {{
            outData = inData[gl_InvocationID];
            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

            float uSamples = 0.0;
            ComputeCurveSamples(uSamples); // uses outData and changes outData.deltaU

            uSamples = min(uSamples, gl_MaxTessGenLevel);
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = uSamples;
        }}
    """

    curveTEShaderCode = """
        #version 410 core

        layout (isolines, point_mode) in;

        const int header = 2;

        patch in SplineInfo
        {
            int uOrder;
            int uN;
            int uM;
            float firstU;
            float lastU;
            float deltaU;
        } inData;

        out SplineInfo
        {
            int uOrder;
            int uN;
            int uM;
            float firstU;
            float lastU;
            float deltaU;
        } outData;

        void main()
        {
            outData.uOrder = inData.uOrder;
            outData.uN = inData.uN;
            outData.uM = inData.uM;
            // gl_TessCoord.x is in [0.0, 1.0]
            float uInterval = inData.lastU - inData.firstU;
            outData.deltaU = uInterval  / gl_TessLevelOuter[1];
            outData.firstU = inData.firstU + gl_TessCoord.x * uInterval;
            // Ensure outData.lastU is in [inData.firstU, inData.lastU].
            outData.lastU = min(inData.lastU, outData.firstU + outData.deltaU);
        }
    """

    curveGeometryShaderCode = """
        #version 410 core

        layout( points ) in;
        layout( line_strip, max_vertices = {maxVertices} ) out;

        const int header = 2;

        in SplineInfo
        {{
            int uOrder;
            int uN;
            int uM;
            float firstU;
            float lastU;
            float deltaU;
        }} inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec3 uSplineColor;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor;

        {computeBasisCode}

        void DrawCurvePoints(in int order, in int n, in int m, in float u)
        {{
            float uBasis[{maxBasis}];
            float duBasis[{maxBasis}];

            ComputeBasis(header, order, n, m, u, uBasis, duBasis);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            vec3 dPoint = vec3(0.0, 0.0, 0.0);
            int i = header + order + n + 4 * (m + 1 - order);
            for (int b = 0; b < order; b++) // loop from coefficient[m+1-order] to coefficient[m+1]
            {{
                point.x += uBasis[b] * texelFetch(uSplineData, i).x;
                point.y += uBasis[b] * texelFetch(uSplineData, i+1).x;
                point.z += uBasis[b] * texelFetch(uSplineData, i+2).x;
                point.w += uBasis[b] * texelFetch(uSplineData, i+3).x;
                dPoint.x += duBasis[b] * texelFetch(uSplineData, i).x;
                dPoint.y += duBasis[b] * texelFetch(uSplineData, i+1).x;
                dPoint.z += duBasis[b] * texelFetch(uSplineData, i+2).x;
                i += 4;
            }}

            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            gl_Position = uProjectionMatrix * vec4(point.x - 0.01*dPoint.y, point.y + 0.01*dPoint.x, point.z, point.w);
            EmitVertex();
            gl_Position = uProjectionMatrix * point;
            EmitVertex();
        }}

        void main()
        {{
            if (inData[0].firstU >= inData[0].lastU)
            {{
                return;
            }}
            int order = inData[0].uOrder;
            int n = inData[0].uN;
            int m = inData[0].uM;
            float u = inData[0].firstU;
            float lastU = inData[0].lastU;
            float deltaU = inData[0].deltaU;
            int vertices = 0;

            splineColor = uSplineColor;
            while (u < lastU && vertices <= {maxVertices} - 6) // Save room for the vertices at u and lastU
            {{
                DrawCurvePoints(order, n, m, u);
                u += deltaU;
                vertices += 3;
            }}
            DrawCurvePoints(order, n, m, lastU);
            vertices += 3;
            EndPrimitive();
        }}
    """

    surfaceVertexShaderCode = """
        #version 410 core

        const int header = 4;
     
        attribute vec4 aParameters;

        uniform samplerBuffer uSplineData;

        out SplineInfo
        {
            int uOrder;
            int vOrder;
            int uN;
            int vN;
            int uM;
            int vM;
            float firstU;
            float firstV;
            float lastU;
            float lastV;
            float deltaU;
            float deltaV;
        } outData;

        void main()
        {
            outData.uOrder = int(texelFetch(uSplineData, 0).x);
            outData.vOrder = int(texelFetch(uSplineData, 1).x);
            outData.uN = int(texelFetch(uSplineData, 2).x);
            outData.vN = int(texelFetch(uSplineData, 3).x);
            int stride = outData.vN - outData.vOrder + 1;

            outData.uM = min(int(gl_InstanceID / stride) + outData.uOrder - 1, outData.uN - 1);
            outData.vM = min(int(mod(gl_InstanceID, stride)) + outData.vOrder - 1, outData.vN - 1);
            outData.firstU = texelFetch(uSplineData, header + outData.uM).x; // uKnots[uM]
            outData.firstV = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vM).x; // vKnots[vM]
            outData.lastU = texelFetch(uSplineData, header + outData.uM + 1).x; // uKnots[uM+1]
            outData.lastV = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vM + 1).x; // vKnots[vM+1]
            outData.deltaU = outData.lastU - outData.firstU;
            outData.deltaV = outData.lastV - outData.firstV;
            gl_Position = aParameters;
        }
    """

    computeSurfaceDeltasCode = """
        void ComputeSurfaceDeltas(in int uOrder, in int uN, in int uM,
            in int vOrder, in int vN, in int vM, out float deltaU, out float deltaV)
        {
            float uInterval = texelFetch(uSplineData, header + uM+1).x - texelFetch(uSplineData, header + uM).x; // uKnots[uM+1] - uKnots[uM]
            float vInterval = texelFetch(uSplineData, header + uOrder+uN + vM+1).x - texelFetch(uSplineData, header + uOrder+uN + vM).x; // vKnots[vM+1] - vKnots[vM]

            deltaU = max(uInterval, 1.0e-8);
            for (int j = vM+1-vOrder; j < vM+1; j++)
            {
                int i = uM+1-uOrder;
                int coefficientOffset = header + uOrder+uN + vOrder+vN + 4*vN*i + 4*j;
                vec4 coefficient0 = vec4(
                    texelFetch(uSplineData, coefficientOffset+0).x, 
                    texelFetch(uSplineData, coefficientOffset+1).x,
                    texelFetch(uSplineData, coefficientOffset+2).x,
                    texelFetch(uSplineData, coefficientOffset+3).x);
                coefficientOffset += 4*vN;
                vec4 coefficient1 = vec4(
                    texelFetch(uSplineData, coefficientOffset+0).x, 
                    texelFetch(uSplineData, coefficientOffset+1).x,
                    texelFetch(uSplineData, coefficientOffset+2).x,
                    texelFetch(uSplineData, coefficientOffset+3).x);
                float gap = texelFetch(uSplineData, header + i+uOrder).x - texelFetch(uSplineData, header + i+1).x; // uKnots[i+uOrder] - uKnots[i+1]
                gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                vec3 dPoint0 = (uOrder - 1) * gap * (coefficient1.xyz - coefficient0.xyz);
                while (i < uM-1)
                {
                    coefficientOffset += 4*vN;
                    vec4 coefficient2 = vec4(
                        texelFetch(uSplineData, coefficientOffset+0).x, 
                        texelFetch(uSplineData, coefficientOffset+1).x,
                        texelFetch(uSplineData, coefficientOffset+2).x,
                        texelFetch(uSplineData, coefficientOffset+3).x);
                    gap = texelFetch(uSplineData, header + i+1+uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+1+uOrder] - uKnots[i+2]
                    gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                    vec3 dPoint1 = (uOrder - 1) * gap * (coefficient2.xyz - coefficient1.xyz);
                    gap = texelFetch(uSplineData, header + i+uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+uOrder] - uKnots[i+2]
                    gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                    vec3 d2Point = (uOrder - 2) * gap * (dPoint1 - dPoint0);

                    ComputeDelta(coefficient0, dPoint0, d2Point, deltaU);
                    ComputeDelta(coefficient1, dPoint0, d2Point, deltaU);
                    ComputeDelta(coefficient1, dPoint1, d2Point, deltaU);
                    ComputeDelta(coefficient2, dPoint1, d2Point, deltaU);

                    coefficient0 = coefficient1;
                    coefficient1 = coefficient2;
                    dPoint0 = dPoint1;
                    i++;
                }
            }

            deltaV = max(vInterval, 1.0e-8);
            for (int i = uM+1-uOrder; i < uM+1; i++)
            {
                int j = vM+1-vOrder;
                int coefficientOffset = header + uOrder+uN + vOrder+vN + 4*vN*i + 4*j;
                vec4 coefficient0 = vec4(
                    texelFetch(uSplineData, coefficientOffset+0).x, 
                    texelFetch(uSplineData, coefficientOffset+1).x,
                    texelFetch(uSplineData, coefficientOffset+2).x,
                    texelFetch(uSplineData, coefficientOffset+3).x);
                coefficientOffset += 4;
                vec4 coefficient1 = vec4(
                    texelFetch(uSplineData, coefficientOffset+0).x, 
                    texelFetch(uSplineData, coefficientOffset+1).x,
                    texelFetch(uSplineData, coefficientOffset+2).x,
                    texelFetch(uSplineData, coefficientOffset+3).x);
                float gap = texelFetch(uSplineData, header + uOrder+uN + j+vOrder).x - texelFetch(uSplineData, header + uOrder+uN + j+1).x; // vKnots[j+vOrder] - vKnots[j+1]
                gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                vec3 dPoint0 = (vOrder - 1) * gap * (coefficient1.xyz - coefficient0.xyz);
                while (j < vM-1)
                {
                    coefficientOffset += 4;
                    vec4 coefficient2 = vec4(
                        texelFetch(uSplineData, coefficientOffset+0).x, 
                        texelFetch(uSplineData, coefficientOffset+1).x,
                        texelFetch(uSplineData, coefficientOffset+2).x,
                        texelFetch(uSplineData, coefficientOffset+3).x);
                    gap = texelFetch(uSplineData, header + uOrder+uN + j+1+vOrder).x - texelFetch(uSplineData, header + uOrder+uN + j+2).x; // vKnots[j+1+vOrder] - vKnots[j+2]
                    gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                    vec3 dPoint1 = (vOrder - 1) * gap * (coefficient2.xyz - coefficient1.xyz);
                    gap = texelFetch(uSplineData, header + uOrder+uN + j+vOrder).x - texelFetch(uSplineData, header + uOrder+uN + j+2).x; // vKnots[j+vOrder] - vKnots[j+2]
                    gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                    vec3 d2Point = (vOrder - 2) * gap * (dPoint1 - dPoint0);

                    ComputeDelta(coefficient0, dPoint0, d2Point, deltaV);
                    ComputeDelta(coefficient1, dPoint0, d2Point, deltaV);
                    ComputeDelta(coefficient1, dPoint1, d2Point, deltaV);
                    ComputeDelta(coefficient2, dPoint1, d2Point, deltaV);

                    coefficient0 = coefficient1;
                    coefficient1 = coefficient2;
                    dPoint0 = dPoint1;
                    j++;
                }
            }
            // uSamples = ceil(uInterval / deltaU);
            // vSamples = ceil(vInterval / deltaV);
        }
    """

    surfaceTCShaderCode = """
        #version 410 core

        layout (vertices = 1) out;

        const int header = 4;

        in SplineInfo
        {{
            int uOrder;
            int vOrder;
            int uN;
            int vN;
            int uM;
            int vM;
            float firstU;
            float firstV;
            float lastU;
            float lastV;
            float deltaU;
            float deltaV;
        }} inData[];

        uniform vec3 uScreenScale;
        uniform samplerBuffer uSplineData;

        patch out SplineInfo
        {{
            int uOrder;
            int vOrder;
            int uN;
            int vN;
            int uM;
            int vM;
            float firstU;
            float firstV;
            float lastU;
            float lastV;
            float deltaU;
            float deltaV;
        }} outData;

        void main()
        {{
            outData = inData[gl_InvocationID];
            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

            // Find bounding box for coefficients in projected screen space pixels.
            vec2 minProjected = uScreenScale.xy;
            vec2 maxProjected = -uScreenScale.xy;
            int j = header + outData.uOrder + outData.uN + outData.vOrder + outData.vN + (outData.vM + 1 - outData.vOrder) * 4;
            for (int vB = 0; vB < outData.vOrder; vB++)
            {{
                int i = j + (outData.uM + 1 - outData.uOrder) * outData.vN * 4;
                for (int uB = 0; uB < outData.uOrder; uB++)
                {{
                    float zScale = uScreenScale.z > 1.0 ? 1.0 / (uScreenScale.z - texelFetch(uSplineData, i+2).x) : 1.0;
                    vec2 projection = vec2(uScreenScale.x * texelFetch(uSplineData, i).x * zScale,
                        uScreenScale.y * texelFetch(uSplineData, i+1).x * zScale);
                    minProjected = min(minProjected, projection);
                    maxProjected = max(maxProjected, projection);
                    i += outData.vN * 4;
                }}
                j += 4;
            }}
            minProjected = max(minProjected, -uScreenScale.xy);
            maxProjected = min(maxProjected, uScreenScale.xy);

            // Calculate the tessellation levels based on one vertex per pixel, so each sample can handle at most "maxVertices" pixels.
            // The number of samples ranges from 1 to gl_MaxTessGenLevel.
            vec2 size = maxProjected - minProjected;
            float samples = min(ceil(0.5 + max(size.x, size.y) / {maxVertices}.0), gl_MaxTessGenLevel);
            gl_TessLevelOuter[0] = samples;
            gl_TessLevelOuter[1] = samples;
        }}
    """

    surfaceTEShaderCode = """
        #version 410 core

        layout (isolines, point_mode) in;

        const int header = 2;

        patch in SplineInfo
        {
            int uOrder;
            int vOrder;
            int uN;
            int vN;
            int uM;
            int vM;
            float firstU;
            float firstV;
            float lastU;
            float lastV;
            float deltaU;
            float deltaV;
        } inData;

        out SplineInfo
        {
            int uOrder;
            int vOrder;
            int uN;
            int vN;
            int uM;
            int vM;
            float firstU;
            float firstV;
            float lastU;
            float lastV;
            float deltaU;
            float deltaV;
        } outData;

        void main()
        {
            outData.uOrder = inData.uOrder;
            outData.vOrder = inData.vOrder;
            outData.uN = inData.uN;
            outData.vN = inData.vN;
            outData.uM = inData.uM;
            outData.vM = inData.vM;

            // gl_TessCoord.x is in [0.0, 1.0]
            float deltaU = inData.lastU - inData.firstU;
            outData.firstU = inData.firstU + gl_TessCoord.x * deltaU;
            // Ensure outData.lastU is in [inData.firstU, inData.lastU].
            outData.lastU = min(inData.lastU, outData.firstU + deltaU / gl_TessLevelOuter[1]);

            // gl_TessCoord.y is in [0.0, 1.0)
            float deltaV = inData.lastV - inData.firstV;
            outData.firstV = inData.firstV +  gl_TessCoord.y * deltaV;
            // Ensure outData.lastV is in [inData.firstV, inData.lastV].
            outData.lastV = min(inData.lastV, outData.firstV + deltaV / gl_TessLevelOuter[0]);
        }
    """

    surfaceGeometryShaderCode = """
        #version 410 core

        layout( points ) in;
        layout( triangle_strip, max_vertices = {maxVertices} ) out;

        const int header = 4;

        in SplineInfo
        {{
            int uOrder;
            int vOrder;
            int uN;
            int vN;
            int uM;
            int vM;
            float firstU;
            float firstV;
            float lastU;
            float lastV;
            float deltaU;
            float deltaV;
        }} inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec3 uSplineColor;
        uniform vec3 uLightDirection;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor;

        {computeBasisCode}

        {computeDeltaCode}

        {computeSurfaceDeltasCode}

        void DrawSurfacePoints(in int uOrder, in int uN, in int uM,
            in float uBasis[{maxBasis}], in float duBasis[{maxBasis}],
            in float uBasisNext[{maxBasis}], in float duBasisNext[{maxBasis}],
            in int vOrder, in int vN, in int vM, in float v)
        {{
            float vBasis[{maxBasis}];
            float dvBasis[{maxBasis}];

            ComputeBasis(header + uOrder + uN, vOrder, vN, vM, v, vBasis, dvBasis);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            vec3 duPoint = vec3(0.0, 0.0, 0.0);
            vec3 dvPoint = vec3(0.0, 0.0, 0.0);
            int i = header + uOrder + uN + vOrder + vN + (uM + 1 - uOrder) * vN * 4;
            for (int uB = 0; uB < uOrder; uB++)
            {{
                int j = i + (vM + 1 - vOrder) * 4;
                for (int vB = 0; vB < vOrder; vB++)
                {{
                    point.x += uBasis[uB] * vBasis[vB] * texelFetch(uSplineData, j).x;
                    point.y += uBasis[uB] * vBasis[vB] * texelFetch(uSplineData, j+1).x;
                    point.z += uBasis[uB] * vBasis[vB] * texelFetch(uSplineData, j+2).x;
                    point.w += uBasis[uB] * vBasis[vB] * texelFetch(uSplineData, j+3).x;
                    duPoint.x += duBasis[uB] * vBasis[vB] * texelFetch(uSplineData, j).x;
                    duPoint.y += duBasis[uB] * vBasis[vB] * texelFetch(uSplineData, j+1).x;
                    duPoint.z += duBasis[uB] * vBasis[vB] * texelFetch(uSplineData, j+2).x;
                    dvPoint.x += uBasis[uB] * dvBasis[vB] * texelFetch(uSplineData, j).x;
                    dvPoint.y += uBasis[uB] * dvBasis[vB] * texelFetch(uSplineData, j+1).x;
                    dvPoint.z += uBasis[uB] * dvBasis[vB] * texelFetch(uSplineData, j+2).x;
                    j += 4;
                }}
                i += vN * 4;
            }}

            vec3 normal = normalize(cross(duPoint, dvPoint));
            float intensity = dot(normal, uLightDirection);
            splineColor = (0.3 + 0.7 * abs(intensity)) * uSplineColor;
            gl_Position = uProjectionMatrix * point;
            EmitVertex();
            
            point = vec4(0.0, 0.0, 0.0, 0.0);
            duPoint = vec3(0.0, 0.0, 0.0);
            dvPoint = vec3(0.0, 0.0, 0.0);
            i = header + uOrder + uN + vOrder + vN + (uM + 1 - uOrder) * vN * 4;
            for (int uB = 0; uB < uOrder; uB++)
            {{
                int j = i + (vM + 1 - vOrder) * 4;
                for (int vB = 0; vB < vOrder; vB++)
                {{
                    point.x += uBasisNext[uB] * vBasis[vB] * texelFetch(uSplineData, j).x;
                    point.y += uBasisNext[uB] * vBasis[vB] * texelFetch(uSplineData, j+1).x;
                    point.z += uBasisNext[uB] * vBasis[vB] * texelFetch(uSplineData, j+2).x;
                    point.w += uBasisNext[uB] * vBasis[vB] * texelFetch(uSplineData, j+3).x;
                    duPoint.x += duBasisNext[uB] * vBasis[vB] * texelFetch(uSplineData, j).x;
                    duPoint.y += duBasisNext[uB] * vBasis[vB] * texelFetch(uSplineData, j+1).x;
                    duPoint.z += duBasisNext[uB] * vBasis[vB] * texelFetch(uSplineData, j+2).x;
                    dvPoint.x += uBasisNext[uB] * dvBasis[vB] * texelFetch(uSplineData, j).x;
                    dvPoint.y += uBasisNext[uB] * dvBasis[vB] * texelFetch(uSplineData, j+1).x;
                    dvPoint.z += uBasisNext[uB] * dvBasis[vB] * texelFetch(uSplineData, j+2).x;
                    j += 4;
                }}
                i += vN * 4;
            }}

            normal = normalize(cross(duPoint, dvPoint));
            intensity = dot(normal, uLightDirection);
            splineColor = (0.3 + 0.7 * abs(intensity)) * uSplineColor;
            gl_Position = uProjectionMatrix * point;
            EmitVertex();
        }}

        void main()
        {{
            if (inData[0].firstU >= inData[0].lastU || inData[0].firstV >= inData[0].lastV)
            {{
                return;
            }}
            int uOrder = inData[0].uOrder;
            int vOrder = inData[0].vOrder;
            int uN = inData[0].uN;
            int vN = inData[0].vN;
            int uM = inData[0].uM;
            int vM = inData[0].vM;
            float firstU = inData[0].firstU;
            float lastU = inData[0].lastU;
            float u = firstU;
            int vertices = 0;
            float uBasis[{maxBasis}];
            float duBasis[{maxBasis}];
            float deltaU;
            float deltaV;

            ComputeSurfaceDeltas(uOrder, uN, uM, vOrder, vN, vM, deltaU, deltaV);
            splineColor = uSplineColor;
            
            ComputeBasis(header, uOrder, uN, uM, u, uBasis, duBasis);
            while (u < lastU && vertices < {maxVertices})
            {{
                u = min(u + deltaU, lastU);
                float uBasisNext[{maxBasis}];
                float duBasisNext[{maxBasis}];
                ComputeBasis(header, uOrder, uN, uM, u, uBasisNext, duBasisNext);

                float v = inData[0].firstV;
                float lastV = inData[0].lastV;
                while (v < lastV && vertices <= {maxVertices} - 4) // Save room for the vertices at v and lastV
                {{
                    DrawSurfacePoints(uOrder, uN, uM, uBasis, duBasis, uBasisNext, duBasisNext,
                        vOrder, vN, vM, v);
                    v = min(v + deltaV, lastV);
                    vertices += 2;
                }}
                DrawSurfacePoints(uOrder, uN, uM, uBasis, duBasis, uBasisNext, duBasisNext,
                    vOrder, vN, vM, v);
                vertices += 2;
                EndPrimitive();
                uBasis = uBasisNext;
                duBasis = duBasisNext;
            }}
        }}
    """

    fragmentShaderCode = """
        #version 410 core
     
        in vec3 splineColor;
        out vec3 color;
     
        void main()
        {
            color = splineColor;
        }
    """
 
    def __init__(self, *args, **kw):
        OpenGLFrame.__init__(self, *args, **kw)
        self.animate = 0 # Set to number of milliseconds before showing next frame (0 means no animation)
        self.splineDrawList = []
        self.currentQ = quat.one
        self.lastQ = quat.one
        self.origin = None
        self.bind("<ButtonPress-3>", self.Home)
        self.bind("<ButtonPress-1>", self.RotateStartHandler)
        self.bind("<ButtonRelease-1>", self.RotateEndHandler)
        self.bind("<B1-Motion>", self.RotateDragHandler)
        self.glInitialized = False

    def initgl(self):
        if not self.glInitialized:
            self.maxVertices = glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES)
            #print("GL_VERSION: ", glGetString(GL_VERSION))
            #print("GL_SHADING_LANGUAGE_VERSION: ", glGetString(GL_SHADING_LANGUAGE_VERSION))
            #print("GL_MAX_GEOMETRY_OUTPUT_VERTICES: ", glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES))
            #print("GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS: ", glGetIntegerv(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS))
            #print("GL_MAX_TESS_GEN_LEVEL: ", glGetIntegerv(GL_MAX_TESS_GEN_LEVEL))

            try:
                self.computeBasisCode = self.computeBasisCode.format(maxBasis=Spline.maxOrder+1)

                self.curveTCShaderCode = self.curveTCShaderCode.format(maxVertices=self.maxVertices,
                    computeDeltaCode=self.computeDeltaCode,
                    computeCurveSamplesCode=self.computeCurveSamplesCode)
                self.curveGeometryShaderCode = self.curveGeometryShaderCode.format(maxVertices=self.maxVertices,
                    computeBasisCode=self.computeBasisCode,
                    maxBasis=Spline.maxOrder+1)
                self.curveProgram = shaders.compileProgram(
                    shaders.compileShader(self.curveVertexShaderCode, GL_VERTEX_SHADER), 
                    shaders.compileShader(self.curveTCShaderCode, GL_TESS_CONTROL_SHADER), 
                    shaders.compileShader(self.curveTEShaderCode, GL_TESS_EVALUATION_SHADER), 
                    shaders.compileShader(self.curveGeometryShaderCode, GL_GEOMETRY_SHADER), 
                    shaders.compileShader(self.fragmentShaderCode, GL_FRAGMENT_SHADER))
                
                self.surfaceTCShaderCode = self.surfaceTCShaderCode.format(maxVertices=self.maxVertices)
                self.surfaceGeometryShaderCode = self.surfaceGeometryShaderCode.format(maxVertices=self.maxVertices,
                    computeBasisCode=self.computeBasisCode,
                    computeDeltaCode=self.computeDeltaCode,
                    computeSurfaceDeltasCode=self.computeSurfaceDeltasCode,
                    maxBasis=Spline.maxOrder+1)
                self.surfaceProgram = shaders.compileProgram(
                    shaders.compileShader(self.surfaceVertexShaderCode, GL_VERTEX_SHADER), 
                    shaders.compileShader(self.surfaceTCShaderCode, GL_TESS_CONTROL_SHADER), 
                    shaders.compileShader(self.surfaceTEShaderCode, GL_TESS_EVALUATION_SHADER), 
                    shaders.compileShader(self.surfaceGeometryShaderCode, GL_GEOMETRY_SHADER), 
                    shaders.compileShader(self.fragmentShaderCode, GL_FRAGMENT_SHADER))
            except shaders.ShaderCompilationError as exception:
                error = exception.args[0]
                lineNumber = error.split(":")[3]
                source = exception.args[1][0]
                badLine = source.split(b"\n")[int(lineNumber)-1]
                shaderType = exception.args[2]
                print(shaderType, error)
                print(badLine)
                quit()

            self.splineDataBuffer = glGenBuffers(1)
            self.splineTextureBuffer = glGenTextures(1)
            glBindBuffer(GL_TEXTURE_BUFFER, self.splineDataBuffer)
            glBindTexture(GL_TEXTURE_BUFFER, self.splineTextureBuffer)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.splineDataBuffer)
            maxFloats = 4 + 2 * Spline.maxKnots + 4 * Spline.maxCoefficients * Spline.maxCoefficients
            glBufferData(GL_TEXTURE_BUFFER, 4 * maxFloats, None, GL_STATIC_READ)

            self.parameterBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glBufferData(GL_ARRAY_BUFFER, 4 * 4, np.array([0,0,0,0], np.float32), GL_STATIC_DRAW)

            glUseProgram(self.curveProgram)
            self.aCurveParameters = glGetAttribLocation(self.curveProgram, "aParameters")
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glVertexAttribPointer(self.aCurveParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uCurveProjectionMatrix = glGetUniformLocation(self.curveProgram, 'uProjectionMatrix')
            self.uCurveScreenScale = glGetUniformLocation(self.curveProgram, 'uScreenScale')
            self.uCurveSplineColor = glGetUniformLocation(self.curveProgram, 'uSplineColor')
            self.uCurveSplineData = glGetUniformLocation(self.curveProgram, 'uSplineData')
            glUniform1i(self.uCurveSplineData, 0) # 0 is the active texture (default is 0)

            glUseProgram(self.surfaceProgram)
            self.aSurfaceParameters = glGetAttribLocation(self.surfaceProgram, "aParameters")
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glVertexAttribPointer(self.aSurfaceParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uSurfaceProjectionMatrix = glGetUniformLocation(self.surfaceProgram, 'uProjectionMatrix')
            self.uSurfaceScreenScale = glGetUniformLocation(self.surfaceProgram, 'uScreenScale')
            self.uSurfaceSplineColor = glGetUniformLocation(self.surfaceProgram, 'uSplineColor')
            self.uSurfaceLightDirection = glGetUniformLocation(self.surfaceProgram, 'uLightDirection')
            self.lightDirection = np.array((0.6, 0.3, 1), np.float32)
            self.lightDirection = self.lightDirection / np.linalg.norm(self.lightDirection)
            glUniform3fv(self.uSurfaceLightDirection, 1, self.lightDirection)
            self.uSurfaceSplineData = glGetUniformLocation(self.surfaceProgram, 'uSplineData')
            glUniform1i(self.uSurfaceSplineData, 0) # 0 is the active texture (default is 0)

            glUseProgram(0)

            glEnable( GL_DEPTH_TEST )
            #glClearColor(1.0, 1.0, 1.0, 0.0)
            glClearColor(0.0, 0.0, 0.0, 0.0)

            self.glInitialized = True

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        xExtent = self.width / self.height
        clipDistance = np.sqrt(3.0)
        zDropoff = 3.0
        near = zDropoff - clipDistance
        far = zDropoff + clipDistance
        top = clipDistance * near / zDropoff # Choose frustum that displays [-clipDistance,clipDistance] in y for z = -zDropoff
        glFrustum(-top*xExtent, top*xExtent, -top, top, near, far)
        glTranslate(0.0, 0.0, -zDropoff)
        #glOrtho(-xExtent, xExtent, -1.0, 1.0, -1.0, 1.0)

        self.projection = glGetFloatv(GL_PROJECTION_MATRIX)
        self.screenScale = np.array((0.5 * self.height * self.projection[0,0], 0.5 * self.height * self.projection[1,1], self.projection[3,3]), np.float32)
        glUseProgram(self.curveProgram)
        glUniformMatrix4fv(self.uCurveProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uCurveScreenScale, 1, self.screenScale)
        glUseProgram(self.surfaceProgram)
        glUniformMatrix4fv(self.uSurfaceProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uSurfaceScreenScale, 1, self.screenScale)
        glUseProgram(0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def redraw(self):

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glLoadIdentity()
        rotation33 = quat.as_rotation_matrix(self.currentQ * self.lastQ)
        rotation44 = np.identity(4, np.float32)
        rotation44[0:3,0:3] = rotation33.T # Transpose to match OpenGL format in numpy
        transform = rotation44

        for spline in self.splineDrawList:
            spline.Draw(self, transform)

        glFlush()

    def Home(self, event):
        self.lastQ = quat.one
        self.currentQ = quat.one
        self.tkExpose(None)

    def ProjectToSphere(self, point):
        length = np.linalg.norm(point)
        if length <= 0.7071: # 1/sqrt(2)
            projection = np.array((point[0], point[1], np.sqrt(1.0 - length * length)), np.float32)
        else:
            projection = np.array((point[0], point[1], 0.5 / length), np.float32)
            projection = projection / np.linalg.norm(projection)
        return projection

    def RotateStartHandler(self, event):
        self.origin = np.array(((2.0 * event.x - self.width)/self.height, (self.height - 2.0 * event.y)/self.height), np.float32)

    def RotateDragHandler(self, event):
        if self.origin is not None:
            point = np.array(((2.0 * event.x - self.width)/self.height, (self.height - 2.0 * event.y)/self.height), np.float32)
            a = self.ProjectToSphere(self.origin)
            b = self.ProjectToSphere(point)
            dot = np.dot(a, b)
            halfCosine = np.sqrt(0.5 * (1.0 + dot))
            halfSine = np.sqrt(0.5 * (1.0 - dot))
            n = np.cross(a,b)
            if halfSine > 1.0e-8:
                n = (halfSine / np.linalg.norm(n)) * n
            self.currentQ = quat.from_float_array((halfCosine, n[0], n[1], n[2]))
            self.tkExpose(None)

    def RotateEndHandler(self, event):
        if self.origin is not None:
            self.lastQ = self.currentQ * self.lastQ
            self.currentQ = quat.one
            self.origin = None