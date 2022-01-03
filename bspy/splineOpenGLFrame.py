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

            for (int i = 0; i < {maxBasis}; i++)
            {{
                uBasis[i] = 0.0;
                duBasis[i] = 0.0;
            }}
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
            if (degree < order)
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
            }}
        }}
    """

    computeSampleRateCode = """
        float ComputeSampleRate(in vec4 point, in vec3 dPoint, in vec3 d2Point, in float minRate)
        {
            float rate = 0.0;
            float scale = uScreenScale.z > 0.0 ? -point.z : 1.0;

            if (point.z < uClipBounds[3] && point.z > uClipBounds[2] && 
                point.y < scale * uClipBounds[1] && point.y > -scale * uClipBounds[1] &&
                point.x < scale * uClipBounds[0] && point.x > -scale * uClipBounds[0])
            {
                float zScale = -1.0 / point.z;
                float zScale2 = zScale * zScale;
                float zScale3 = zScale2 * zScale;
                vec2 projection = uScreenScale.z > 0.0 ? 
                    vec2(uScreenScale.x * (d2Point.x * zScale - 2.0 * dPoint.x * dPoint.z * zScale2 +
                        point.x * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2)),
                        uScreenScale.y * (d2Point.y * zScale - 2.0 * dPoint.y * dPoint.z * zScale2 +
                        point.y * (2.0 * dPoint.z * dPoint.z * zScale3 - d2Point.z * zScale2)))
                    : vec2(uScreenScale.x * d2Point.x, uScreenScale.y * d2Point.y);
                rate = max(sqrt(length(projection)), minRate);
            }
            return rate;
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
            float u;
            float uInterval;
        } outData;

        void main()
        {
            outData.uOrder = int(texelFetch(uSplineData, 0).x);
            outData.uN = int(texelFetch(uSplineData, 1).x);
            outData.uM = min(gl_InstanceID + outData.uOrder - 1, outData.uN - 1);
            outData.u = texelFetch(uSplineData, header + outData.uM).x; // knots[uM]
            outData.uInterval = texelFetch(uSplineData, header + outData.uM + 1).x - outData.u; // knots[uM+1] - knots[uM]
            gl_Position = aParameters;
        }
    """

    computeCurveSamplesCode = """
        void ComputeCurveSamples(out float uSamples)
        {
            float sampleRate = 0.0;
            if (outData.uInterval > 0.0)
            {
                float minRate = 1.0 / outData.uInterval;
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

                    sampleRate = max(sampleRate, ComputeSampleRate(coefficient0, dPoint0, d2Point, minRate));
                    sampleRate = max(sampleRate, ComputeSampleRate(coefficient1, dPoint0, d2Point, minRate));
                    sampleRate = max(sampleRate, ComputeSampleRate(coefficient1, dPoint1, d2Point, minRate));
                    sampleRate = max(sampleRate, ComputeSampleRate(coefficient2, dPoint1, d2Point, minRate));

                    coefficient0 = coefficient1;
                    coefficient1 = coefficient2;
                    dPoint0 = dPoint1;
                    i++;
                }
            }
            uSamples = min(floor(0.5 + outData.uInterval * sampleRate), gl_MaxTessGenLevel);
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
            float u;
            float uInterval;
        }} inData[];

        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform samplerBuffer uSplineData;

        patch out SplineInfo
        {{
            int uOrder;
            int uN;
            int uM;
            float u;
            float uInterval;
        }} outData;

        {computeSampleRateCode}

        {computeCurveSamplesCode}

        void main()
        {{
            outData = inData[gl_InvocationID];
            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

            float uSamples = 0.0;
            ComputeCurveSamples(uSamples);
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = uSamples;
        }}
    """

    curveTEShaderCode = """
        #version 410 core

        layout (isolines) in;

        const int header = 2;

        patch in SplineInfo
        {{
            int uOrder;
            int uN;
            int uM;
            float u;
            float uInterval;
        }} inData;

        uniform mat4 uProjectionMatrix;
        uniform samplerBuffer uSplineData;

        {computeBasisCode}

        void main()
        {{
            float uBasis[{maxBasis}];
            float duBasis[{maxBasis}];
            ComputeBasis(header, inData.uOrder, inData.uN, inData.uM,
                inData.u + gl_TessCoord.x * inData.uInterval, 
                uBasis, duBasis);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            int i = header + inData.uOrder + inData.uN + 4 * (inData.uM + 1 - inData.uOrder);
            for (int b = 0; b < inData.uOrder; b++) // loop from coefficient[m+1-order] to coefficient[m+1]
            {{
                point.x += uBasis[b] * texelFetch(uSplineData, i).x;
                point.y += uBasis[b] * texelFetch(uSplineData, i+1).x;
                point.z += uBasis[b] * texelFetch(uSplineData, i+2).x;
                point.w += uBasis[b] * texelFetch(uSplineData, i+3).x;
                i += 4;
            }}

            gl_Position = uProjectionMatrix * point;
        }}
    """

    curveFragmentShaderCode = """
        #version 410 core
     
        uniform vec4 uLineColor;

        out vec4 color;
     
        void main()
        {
            color = uLineColor;
        }
    """

    surfaceVertexShaderCode = """
        #version 410 core

        const int header = 4;
     
        attribute vec4 aParameters;

        uniform samplerBuffer uSplineData;

        out SplineInfo
        {
            int uOrder, vOrder;
            int uN, vN;
            int uM, vM;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
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
            outData.uFirst = texelFetch(uSplineData, header + outData.uOrder - 1).x; // uKnots[uOrder-1]
            outData.vFirst = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vOrder - 1).x; // vKnots[vOrder-1]
            outData.uSpan = texelFetch(uSplineData, header + outData.uN - 1).x - outData.uFirst; // uKnots[uN-1] - uKnots[uOrder-1]
            outData.vSpan = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vN - 1).x - outData.vFirst; // vKnots[vN-1] - vKnots[vOrder-1]
            outData.u = texelFetch(uSplineData, header + outData.uM).x; // uKnots[uM]
            outData.v = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vM).x; // vKnots[vM]
            outData.uInterval = texelFetch(uSplineData, header + outData.uM + 1).x - outData.u; // uKnots[uM+1] - uKnots[uM]
            outData.vInterval = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vM + 1).x - outData.v; // vKnots[vM+1] - vKnots[vM]
            gl_Position = aParameters;
        }
    """

    computeSurfaceSamplesCode = """
        void ComputeSurfaceSamples(out float uSamples[3], out float vSamples[3])
        {{
            // Computes sample counts for u and v for the left side ([0]), middle ([1]), and right side ([2]).
            // The left side sample count matches the right side sample count for the previous knot.
            // The middle sample count is the number of samples between knots (same as ComputeCurveSamples).
            float sampleRate[3] = float[3](0.0, 0.0, 0.0);
            if (outData.uInterval > 0.0)
            {{
                float sampleRateLeft[{maxOrder}];
                float sampleRateRight[{maxOrder}];
                float minRate = 1.0 / outData.uInterval;

                for (int k = 0; k < outData.uOrder; k++)
                {{
                    sampleRateLeft[k] = 0.0;
                    sampleRateRight[k] = 0.0;
                }}
                for (int j = outData.vM+1-outData.vOrder; j < outData.vM+1; j++)
                {{
                    int i = max(outData.uM - outData.uOrder, 0);
                    int iLimit = min(outData.uM, outData.uN - 2);
                    int coefficientOffset = header + outData.uOrder+outData.uN + outData.vOrder+outData.vN + 4*outData.vN*i + 4*j;
                    vec4 coefficient0 = vec4(
                        texelFetch(uSplineData, coefficientOffset+0).x, 
                        texelFetch(uSplineData, coefficientOffset+1).x,
                        texelFetch(uSplineData, coefficientOffset+2).x,
                        texelFetch(uSplineData, coefficientOffset+3).x);
                    coefficientOffset += 4*outData.vN;
                    vec4 coefficient1 = vec4(
                        texelFetch(uSplineData, coefficientOffset+0).x, 
                        texelFetch(uSplineData, coefficientOffset+1).x,
                        texelFetch(uSplineData, coefficientOffset+2).x,
                        texelFetch(uSplineData, coefficientOffset+3).x);
                    float gap = texelFetch(uSplineData, header + i+outData.uOrder).x - texelFetch(uSplineData, header + i+1).x; // uKnots[i+uOrder] - uKnots[i+1]
                    gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                    vec3 dPoint0 = (outData.uOrder - 1) * gap * (coefficient1.xyz - coefficient0.xyz);
                    while (i < iLimit)
                    {{
                        coefficientOffset += 4*outData.vN;
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

                        int k = i - outData.uM + outData.uOrder;
                        sampleRateLeft[k] = max(sampleRateLeft[k], ComputeSampleRate(coefficient0, dPoint0, d2Point, minRate));
                        sampleRateLeft[k] = max(sampleRateLeft[k], ComputeSampleRate(coefficient1, dPoint0, d2Point, minRate));
                        sampleRateRight[k] = max(sampleRateRight[k], ComputeSampleRate(coefficient1, dPoint1, d2Point, minRate));
                        sampleRateRight[k] = max(sampleRateRight[k], ComputeSampleRate(coefficient2, dPoint1, d2Point, minRate));

                        coefficient0 = coefficient1;
                        coefficient1 = coefficient2;
                        dPoint0 = dPoint1;
                        i++;
                    }}
                }}
                for (int k = 1; k < outData.uOrder - 1; k++)
                {{
                    sampleRate[0] = max(sampleRate[0], sampleRateRight[k-1]);
                    sampleRate[0] = max(sampleRate[0], sampleRateLeft[k]);
                    sampleRate[1] = max(sampleRate[1], sampleRateLeft[k]);
                    sampleRate[1] = max(sampleRate[1], sampleRateRight[k]);
                    sampleRate[2] = max(sampleRate[2], sampleRateRight[k]);
                    sampleRate[2] = max(sampleRate[2], sampleRateLeft[k+1]);
                }}
            }}
            uSamples[0] = min(floor(0.5 + outData.uInterval * sampleRate[0]), gl_MaxTessGenLevel);
            uSamples[1] = min(floor(0.5 + outData.uInterval * sampleRate[1]), gl_MaxTessGenLevel);
            uSamples[2] = min(floor(0.5 + outData.uInterval * sampleRate[2]), gl_MaxTessGenLevel);

            sampleRate = float[3](0.0, 0.0, 0.0);
            if (outData.vInterval > 0.0)
            {{
                float sampleRateLeft[{maxOrder}];
                float sampleRateRight[{maxOrder}];
                float minRate = 1.0 / outData.vInterval;

                for (int k = 0; k < outData.vOrder; k++)
                {{
                    sampleRateLeft[k] = 0.0;
                    sampleRateRight[k] = 0.0;
                }}
                for (int i = outData.uM+1-outData.uOrder; i < outData.uM+1; i++)
                {{
                    int j = max(outData.vM - outData.vOrder, 0);
                    int jLimit = min(outData.vM, outData.vN - 2);
                    int coefficientOffset = header + outData.uOrder+outData.uN + outData.vOrder+outData.vN + 4*outData.vN*i + 4*j;
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
                    float gap = texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+outData.vOrder).x - texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+1).x; // vKnots[j+vOrder] - vKnots[j+1]
                    gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                    vec3 dPoint0 = (outData.vOrder - 1) * gap * (coefficient1.xyz - coefficient0.xyz);
                    while (j < jLimit)
                    {{
                        coefficientOffset += 4;
                        vec4 coefficient2 = vec4(
                            texelFetch(uSplineData, coefficientOffset+0).x, 
                            texelFetch(uSplineData, coefficientOffset+1).x,
                            texelFetch(uSplineData, coefficientOffset+2).x,
                            texelFetch(uSplineData, coefficientOffset+3).x);
                        gap = texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+1+outData.vOrder).x - texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+2).x; // vKnots[j+1+vOrder] - vKnots[j+2]
                        gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                        vec3 dPoint1 = (outData.vOrder - 1) * gap * (coefficient2.xyz - coefficient1.xyz);
                        gap = texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+outData.vOrder).x - texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+2).x; // vKnots[j+vOrder] - vKnots[j+2]
                        gap = gap < 1.0e-8 ? 0.0 : 1.0 / gap;
                        vec3 d2Point = (outData.vOrder - 2) * gap * (dPoint1 - dPoint0);

                        int k = j - outData.vM + outData.vOrder;
                        sampleRateLeft[k] = max(sampleRateLeft[k], ComputeSampleRate(coefficient0, dPoint0, d2Point, minRate));
                        sampleRateLeft[k] = max(sampleRateLeft[k], ComputeSampleRate(coefficient1, dPoint0, d2Point, minRate));
                        sampleRateRight[k] = max(sampleRateRight[k], ComputeSampleRate(coefficient1, dPoint1, d2Point, minRate));
                        sampleRateRight[k] = max(sampleRateRight[k], ComputeSampleRate(coefficient2, dPoint1, d2Point, minRate));

                        coefficient0 = coefficient1;
                        coefficient1 = coefficient2;
                        dPoint0 = dPoint1;
                        j++;
                    }}
                }}
                for (int k = 1; k < outData.vOrder - 1; k++)
                {{
                    sampleRate[0] = max(sampleRate[0], sampleRateRight[k-1]);
                    sampleRate[0] = max(sampleRate[0], sampleRateLeft[k]);
                    sampleRate[1] = max(sampleRate[1], sampleRateLeft[k]);
                    sampleRate[1] = max(sampleRate[1], sampleRateRight[k]);
                    sampleRate[2] = max(sampleRate[2], sampleRateRight[k]);
                    sampleRate[2] = max(sampleRate[2], sampleRateLeft[k+1]);
                }}
            }}
            vSamples[0] = min(floor(0.5 + outData.vInterval * sampleRate[0]), gl_MaxTessGenLevel);
            vSamples[1] = min(floor(0.5 + outData.vInterval * sampleRate[1]), gl_MaxTessGenLevel);
            vSamples[2] = min(floor(0.5 + outData.vInterval * sampleRate[2]), gl_MaxTessGenLevel);
        }}
    """

    surfaceTCShaderCode = """
        #version 410 core

        layout (vertices = 1) out;

        const int header = 4;

        in SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uM, vM;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} inData[];

        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform samplerBuffer uSplineData;

        patch out SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uM, vM;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} outData;

        {computeSampleRateCode}

        {computeSurfaceSamplesCode}

        void main()
        {{
            outData = inData[gl_InvocationID];
            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

            float uSamples[3];
            float vSamples[3];
            ComputeSurfaceSamples(uSamples, vSamples);
            gl_TessLevelOuter[0] = vSamples[0];
            gl_TessLevelOuter[1] = uSamples[0];
            gl_TessLevelOuter[2] = vSamples[2];
            gl_TessLevelOuter[3] = uSamples[2];
            gl_TessLevelInner[0] = uSamples[1];
            gl_TessLevelInner[1] = vSamples[1];
        }}
    """

    surfaceTEShaderCode = """
        #version 410 core

        layout (quads) in;

        const int header = 4;

        patch in SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uM, vM;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} inData;

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform samplerBuffer uSplineData;

        flat out SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uM, vM;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} outData;
        out vec4 worldPosition;
        out vec3 normal;
        out vec2 parameters;
        out vec2 pixelPer;

        {computeBasisCode}

        void main()
        {{
            float uBasis[{maxBasis}];
            float duBasis[{maxBasis}];
            parameters.x = inData.u + gl_TessCoord.x * inData.uInterval;
            ComputeBasis(header, inData.uOrder, inData.uN, inData.uM, parameters.x, uBasis, duBasis);

            float vBasis[{maxBasis}];
            float dvBasis[{maxBasis}];
            parameters.y = inData.v + gl_TessCoord.y * inData.vInterval;
            ComputeBasis(header + inData.uOrder+inData.uN, inData.vOrder, inData.vN, inData.vM, parameters.y, vBasis, dvBasis);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 0.0);
            vec3 duPoint = vec3(0.0, 0.0, 0.0);
            vec3 dvPoint = vec3(0.0, 0.0, 0.0);
            int i = header + inData.uOrder+inData.uN + inData.vOrder+inData.vN + (inData.uM + 1 - inData.uOrder) * inData.vN * 4;
            for (int uB = 0; uB < inData.uOrder; uB++)
            {{
                int j = i + (inData.vM + 1 - inData.vOrder) * 4;
                for (int vB = 0; vB < inData.vOrder; vB++)
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
                i += inData.vN * 4;
            }}

            outData = inData;

            worldPosition = point;
            normal = normalize(cross(duPoint, dvPoint));
            float zScale = 1.0 / point.z * point.z;
            pixelPer.x = zScale * max(uScreenScale.x * abs(point.x * duPoint.z - duPoint.x * point.z), uScreenScale.y * abs(point.y * duPoint.z - duPoint.y * point.z));
            pixelPer.y = zScale * max(uScreenScale.x * abs(point.x * dvPoint.z - dvPoint.x * point.z), uScreenScale.y * abs(point.y * dvPoint.z - dvPoint.y * point.z));
            gl_Position = uProjectionMatrix * point;
        }}
    """

    surfaceFragmentShaderCode = """
        #version 410 core
     
        flat in SplineInfo
        {
            int uOrder, vOrder;
            int uN, vN;
            int uM, vM;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        } inData;
        in vec4 worldPosition;
        in vec3 normal;
        in vec2 parameters;
        in vec2 pixelPer;

        uniform vec4 uFillColor;
        uniform vec4 uLineColor;
        uniform vec3 uLightDirection;
        uniform int uOptions;

        out vec4 color;
     
        void main()
        {
            float specular = pow(abs(dot(normal, normalize(uLightDirection + worldPosition.xyz / length(worldPosition)))), 10.0);
            bool line = (uOptions & (1 << 4)) > 0 && (pixelPer.x * (parameters.x - inData.uFirst) < 12.0 || pixelPer.x * (inData.uFirst + inData.uSpan - parameters.x) < 12.0);
            line = line || ((uOptions & (1 << 4)) > 0 && (pixelPer.y * (parameters.y - inData.vFirst) < 12.0 || pixelPer.y * (inData.vFirst + inData.vSpan - parameters.y) < 12.0));
            line = line || ((uOptions & (1 << 5)) > 0 && pixelPer.x * (parameters.x - inData.u) < 12.0);
            line = line || ((uOptions & (1 << 5)) > 0 && pixelPer.y * (parameters.y - inData.v) < 12.0);
            color = line ? uLineColor : ((uOptions & (1 << 2)) > 0 ? uFillColor : vec4(0.0, 0.0, 0.0, 0.0));
            color.rgb = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * color.rgb;
            if (color.a == 0.0)
                discard;
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
        self.bind("<Unmap>", self.Unmap)

        self.computeBasisCode = self.computeBasisCode.format(maxBasis=Spline.maxOrder+1)
        self.computeSurfaceSamplesCode = self.computeSurfaceSamplesCode.format(maxOrder=Spline.maxOrder)
        self.curveTCShaderCode = self.curveTCShaderCode.format(
            computeSampleRateCode=self.computeSampleRateCode,
            computeCurveSamplesCode=self.computeCurveSamplesCode)
        self.curveTEShaderCode = self.curveTEShaderCode.format(
            computeBasisCode=self.computeBasisCode,
            maxBasis=Spline.maxOrder+1)
        self.surfaceTCShaderCode = self.surfaceTCShaderCode.format(
            computeSampleRateCode=self.computeSampleRateCode,
            computeSurfaceSamplesCode=self.computeSurfaceSamplesCode)
        self.surfaceTEShaderCode = self.surfaceTEShaderCode.format(
            computeBasisCode=self.computeBasisCode,
            maxBasis=Spline.maxOrder+1)

        self.glInitialized = False

    def initgl(self):
        if not self.glInitialized:
            #print("GL_VERSION: ", glGetString(GL_VERSION))
            #print("GL_SHADING_LANGUAGE_VERSION: ", glGetString(GL_SHADING_LANGUAGE_VERSION))
            #print("GL_MAX_TESS_GEN_LEVEL: ", glGetIntegerv(GL_MAX_TESS_GEN_LEVEL))

            try:
                self.curveProgram = shaders.compileProgram(
                    shaders.compileShader(self.curveVertexShaderCode, GL_VERTEX_SHADER), 
                    shaders.compileShader(self.curveTCShaderCode, GL_TESS_CONTROL_SHADER), 
                    shaders.compileShader(self.curveTEShaderCode, GL_TESS_EVALUATION_SHADER), 
                    shaders.compileShader(self.curveFragmentShaderCode, GL_FRAGMENT_SHADER))
                self.surfaceProgram = shaders.compileProgram(
                    shaders.compileShader(self.surfaceVertexShaderCode, GL_VERTEX_SHADER), 
                    shaders.compileShader(self.surfaceTCShaderCode, GL_TESS_CONTROL_SHADER), 
                    shaders.compileShader(self.surfaceTEShaderCode, GL_TESS_EVALUATION_SHADER), 
                    shaders.compileShader(self.surfaceFragmentShaderCode, GL_FRAGMENT_SHADER))
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
            glBufferData(GL_TEXTURE_BUFFER, 4 * Spline.maxFloats, None, GL_STATIC_READ)

            self.parameterBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glBufferData(GL_ARRAY_BUFFER, 4 * 4, np.array([0,0,0,0], np.float32), GL_STATIC_DRAW)

            glUseProgram(self.curveProgram)
            self.aCurveParameters = glGetAttribLocation(self.curveProgram, "aParameters")
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glVertexAttribPointer(self.aCurveParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uCurveProjectionMatrix = glGetUniformLocation(self.curveProgram, 'uProjectionMatrix')
            self.uCurveScreenScale = glGetUniformLocation(self.curveProgram, 'uScreenScale')
            self.uCurveClipBounds = glGetUniformLocation(self.curveProgram, 'uClipBounds')
            self.uCurveLineColor = glGetUniformLocation(self.curveProgram, 'uLineColor')
            self.uCurveSplineData = glGetUniformLocation(self.curveProgram, 'uSplineData')
            glUniform1i(self.uCurveSplineData, 0) # GL_TEXTURE0 is the spline buffer texture

            glUseProgram(self.surfaceProgram)
            self.aSurfaceParameters = glGetAttribLocation(self.surfaceProgram, "aParameters")
            glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
            glVertexAttribPointer(self.aSurfaceParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
            self.uSurfaceProjectionMatrix = glGetUniformLocation(self.surfaceProgram, 'uProjectionMatrix')
            self.uSurfaceScreenScale = glGetUniformLocation(self.surfaceProgram, 'uScreenScale')
            self.uSurfaceClipBounds = glGetUniformLocation(self.curveProgram, 'uClipBounds')
            self.uSurfaceFillColor = glGetUniformLocation(self.surfaceProgram, 'uFillColor')
            self.uSurfaceLineColor = glGetUniformLocation(self.surfaceProgram, 'uLineColor')
            self.uSurfaceLightDirection = glGetUniformLocation(self.surfaceProgram, 'uLightDirection')
            self.lightDirection = np.array((0.6, 0.3, -1.0), np.float32)
            self.lightDirection = self.lightDirection / np.linalg.norm(self.lightDirection)
            glUniform3fv(self.uSurfaceLightDirection, 1, self.lightDirection)
            self.uSurfaceOptions = glGetUniformLocation(self.surfaceProgram, 'uOptions')
            self.uSurfaceSplineData = glGetUniformLocation(self.surfaceProgram, 'uSplineData')
            glUniform1i(self.uSurfaceSplineData, 0) # GL_TEXTURE0 is the spline buffer texture

            glUseProgram(0)

            glEnable( GL_DEPTH_TEST )
            #glClearColor(1.0, 1.0, 1.0, 1.0)
            glClearColor(0.0, 0.0, 0.0, 1.0)

            self.glInitialized = True

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.translation = np.identity(4, np.float32)
        xExtent = self.width / self.height
        clipDistance = np.sqrt(3.0)
        zDropoff = 3.0
        near = zDropoff - clipDistance
        far = zDropoff + clipDistance
        top = clipDistance * near / zDropoff # Choose frustum that displays [-clipDistance,clipDistance] in y for z = -zDropoff
        glFrustum(-top*xExtent, top*xExtent, -top, top, near, far)
        self.translation[3][2] = -zDropoff
        #glOrtho(-xExtent, xExtent, -1.0, 1.0, -1.0, 1.0)

        self.projection = glGetFloatv(GL_PROJECTION_MATRIX)
        self.screenScale = np.array((0.5 * self.height * self.projection[0,0], 0.5 * self.height * self.projection[1,1], 1.0), np.float32)
        self.clipBounds = np.array((1.0 / self.projection[0,0], 1.0 / self.projection[1,1], -far, -near), np.float32)

        glUseProgram(self.curveProgram)
        glUniformMatrix4fv(self.uCurveProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uCurveScreenScale, 1, self.screenScale)
        glUniform4fv(self.uCurveClipBounds, 1, self.clipBounds)
        glUseProgram(self.surfaceProgram)
        glUniformMatrix4fv(self.uSurfaceProjectionMatrix, 1, GL_FALSE, self.projection)
        glUniform3fv(self.uSurfaceScreenScale, 1, self.screenScale)
        glUniform4fv(self.uSurfaceClipBounds, 1, self.clipBounds)
        glUseProgram(0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def redraw(self):

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glLoadIdentity()
        rotation33 = quat.as_rotation_matrix(self.currentQ * self.lastQ)
        rotation44 = np.identity(4, np.float32)
        rotation44[0:3,0:3] = rotation33.T # Transpose to match OpenGL format in numpy
        transform = rotation44 @ self.translation

        for spline in self.splineDrawList:
            spline.Draw(self, transform)

        glFlush()

    def Unmap(self, event):
        self.glInitialized = False

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
            if 1.0e-8 < halfSine < 1.0:
                n = (halfSine / np.linalg.norm(n)) * n
            self.currentQ = quat.from_float_array((halfCosine, n[0], n[1], n[2]))
            self.tkExpose(None)

    def RotateEndHandler(self, event):
        if self.origin is not None:
            self.lastQ = self.currentQ * self.lastQ
            self.currentQ = quat.one
            self.origin = None