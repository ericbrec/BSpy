import numpy as np
import tkinter as tk
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from pyopengltk import OpenGLFrame
from bspy import DrawableSpline
from bspy.drawableSpline import _set_color

class SplineOpenGLFrame(OpenGLFrame):
    """
    A tkinter `OpenGLFrame` with shaders to display a `DrawableSpline` list. 
    """

    ROTATE = 1
    """Default view mode in which dragging the left mouse rotates the view."""
    PAN = 2
    """View mode in which dragging the left mouse pans the view."""
    FLY = 3
    """View mode in which dragging the left mouse flies toward the mouse position."""

    computeBSplineCode = """
        void ComputeBSpline(in int offset, in int order, in int n, in int knot, in float u, 
            out float uBSpline[{maxOrder}], out float duBSpline[{maxOrder}])
        {{
            int degree = 1;

            for (int i = 0; i < {maxOrder}; i++)
            {{
                uBSpline[i] = 0.0;
                duBSpline[i] = 0.0;
            }}
            uBSpline[order-1] = 1.0;

            while (degree < order - 1)
            {{
                int b = order - degree;
                for (int i = knot - degree; i < knot; i++)
                {{
                    float knotValue = texelFetch(uSplineData, offset + i).x; // knots[i]
                    float alpha = (u - knotValue) / (texelFetch(uSplineData, offset + i + degree).x - knotValue); // (u - knots[i]) / (knots[i+degree] - knots[i]);
                    uBSpline[b-1] += (1.0 - alpha) * uBSpline[b];
                    uBSpline[b] *= alpha;
                    b++;
                }}
                degree++;
            }}
            if (degree < order)
            {{
                int b = order - degree;
                for (int i = knot - degree; i < knot; i++)
                {{
                    float knotValue = texelFetch(uSplineData, offset + i).x; // knots[i]
                    float gap = texelFetch(uSplineData, offset + i + degree).x - knotValue; // knots[i+degree] - knots[i]
                    float alpha = degree / gap;
                    duBSpline[b-1] += -alpha * uBSpline[b];
                    duBSpline[b] = alpha * uBSpline[b];

                    alpha = (u - knotValue) / gap; // (u - knots[i]) / gap;
                    uBSpline[b-1] += (1.0 - alpha) * uBSpline[b];
                    uBSpline[b] *= alpha;
                    b++;
                }}
            }}
        }}
    """

    computeSampleRateCode = """
        float ComputeSampleRate(in vec3 point, in vec3 dPoint, in vec3 d2Point, in float minRate)
        {
            float rate = 0.0;
            float scale = uScreenScale.z > 0.0 ? -point.z : 1.0;

            // Only consider points that lie within the clip bounds or whose derivative spans the clip bounds.
            if (point.z < uClipBounds[3] && point.z > uClipBounds[2] && 
                ((point.y < scale * uClipBounds[1] && point.y > -scale * uClipBounds[1]) ||
                (point.y >= scale * uClipBounds[1] && point.y + dPoint.y <= -scale * uClipBounds[1]) ||
                (point.y <= -scale * uClipBounds[1] && point.y + dPoint.y >= scale * uClipBounds[1])) &&
                ((point.x < scale * uClipBounds[0] && point.x > -scale * uClipBounds[0]) ||
                (point.x >= scale * uClipBounds[0] && point.x + dPoint.x <= -scale * uClipBounds[0]) ||
                (point.x <= -scale * uClipBounds[0] && point.x + dPoint.x >= scale * uClipBounds[0])))
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
            int uKnot;
            float u;
            float uInterval;
        } outData;

        void main()
        {
            outData.uOrder = int(texelFetch(uSplineData, 0).x);
            outData.uN = int(texelFetch(uSplineData, 1).x);
            outData.uKnot = min(gl_InstanceID + outData.uOrder, outData.uN);
            outData.u = texelFetch(uSplineData, header + outData.uKnot - 1).x; // knots[uKnot-1]
            outData.uInterval = texelFetch(uSplineData, header + outData.uKnot).x - outData.u; // knots[uKnot] - knots[uKnot-1]
            gl_Position = aParameters;
        }
    """

    computeCurveSamplesCode = """
        void ComputeCurveSamples(in int maxSamples, out float uSamples)
        {
            float sampleRate = 0.0;
            if (outData.uInterval > 0.0)
            {
                float minRate = 1.0 / outData.uInterval;
                if (outData.uOrder < 3)
                {
                    // It's a line or point, so just do the minimum sample.
                    sampleRate = minRate;
                }
                else
                {
                    int i = outData.uKnot - outData.uOrder;
                    int coefficientOffset = header + outData.uOrder + outData.uN + nDep * i;
                    vec3 coefficient0 = vec3(
                        texelFetch(uSplineData, coefficientOffset+0).x, 
                        texelFetch(uSplineData, coefficientOffset+1).x,
                        texelFetch(uSplineData, coefficientOffset+2).x);
                    coefficientOffset += nDep;
                    vec3 coefficient1 = vec3(
                        texelFetch(uSplineData, coefficientOffset+0).x, 
                        texelFetch(uSplineData, coefficientOffset+1).x,
                        texelFetch(uSplineData, coefficientOffset+2).x);
                    float gap = texelFetch(uSplineData, header + i+outData.uOrder).x - texelFetch(uSplineData, header + i+1).x; // uKnots[i+uOrder] - uKnots[i+1]
                    vec3 dPoint0 = ((outData.uOrder - 1) / gap) * (coefficient1 - coefficient0);
                    while (i < outData.uKnot-2)
                    {
                        coefficientOffset += nDep;
                        vec3 coefficient2 = vec3(
                            texelFetch(uSplineData, coefficientOffset+0).x, 
                            texelFetch(uSplineData, coefficientOffset+1).x,
                            texelFetch(uSplineData, coefficientOffset+2).x);
                        gap = texelFetch(uSplineData, header + i+1+outData.uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+1+uOrder] - uKnots[i+2]
                        vec3 dPoint1 = ((outData.uOrder - 1) / gap) * (coefficient2 - coefficient1);
                        gap = texelFetch(uSplineData, header + i+outData.uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+uOrder] - uKnots[i+2]
                        vec3 d2Point = ((outData.uOrder - 2) / gap) * (dPoint1 - dPoint0);

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
            }
            uSamples = min(floor(0.5 + outData.uInterval * sampleRate), maxSamples);
        }
    """

    curveTCShaderCode = """
        #version 410 core

        layout (vertices = 1) out;

        const int header = 2;
        const int nDep = 3;

        in SplineInfo
        {{
            int uOrder;
            int uN;
            int uKnot;
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
            int uKnot;
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
            ComputeCurveSamples(gl_MaxTessGenLevel, uSamples);
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = uSamples;
        }}
    """

    curveTEShaderCode = """
        #version 410 core

        layout (isolines) in;

        const int header = 2;
        const int nDep = 3;

        patch in SplineInfo
        {{
            int uOrder;
            int uN;
            int uKnot;
            float u;
            float uInterval;
        }} inData;

        uniform mat4 uProjectionMatrix;
        uniform samplerBuffer uSplineData;

        {computeBSplineCode}

        void main()
        {{
            float uBSpline[{maxOrder}];
            float duBSpline[{maxOrder}];
            ComputeBSpline(header, inData.uOrder, inData.uN, inData.uKnot,
                inData.u + gl_TessCoord.x * inData.uInterval, 
                uBSpline, duBSpline);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
            int i = header + inData.uOrder + inData.uN + nDep * (inData.uKnot - inData.uOrder);
            for (int b = 0; b < inData.uOrder; b++) // loop from coefficient[uKnot-order] to coefficient[uKnot]
            {{
                point.x += uBSpline[b] * texelFetch(uSplineData, i).x;
                point.y += uBSpline[b] * texelFetch(uSplineData, i+1).x;
                point.z += uBSpline[b] * texelFetch(uSplineData, i+2).x;
                i += nDep;
            }}

            gl_Position = uProjectionMatrix * point;
        }}
    """

    curveGeometryShaderCode = """
        #version 330 core

        layout( points ) in;
        layout( line_strip, max_vertices = 256 ) out;

        const int header = 2;
        const int nDep = 3;

        in SplineInfo
        {{
            int uOrder;
            int uN;
            int uKnot;
            float u;
            float uInterval;
        }} inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform samplerBuffer uSplineData;

        struct SplineInfo
        {{
            int uOrder;
            int uN;
            int uKnot;
            float u;
            float uInterval;
        }} outData; // We don't output outData (too many components per vertex), but we do use it in ComputeCurveSamples.

        {computeSampleRateCode}

        {computeCurveSamplesCode}

        {computeBSplineCode}

        void main()
        {{
            float uSamples = 0.0;

            outData.uOrder = inData[0].uOrder;
            outData.uN = inData[0].uN;
            outData.uKnot = inData[0].uKnot;
            outData.u = inData[0].u;
            outData.uInterval = inData[0].uInterval;
            ComputeCurveSamples(gl_MaxGeometryOutputVertices - 1, uSamples);

            if (uSamples > 0.0)
            {{
                float uBSpline[{maxOrder}];
                float duBSpline[{maxOrder}];
                float u = outData.u;
                float deltaU = outData.uInterval / uSamples;
                int iOffset = header + outData.uOrder + outData.uN + nDep * (outData.uKnot - outData.uOrder);

                for (int uSample = 0; uSample <= uSamples; uSample++)
                {{
                    ComputeBSpline(header, outData.uOrder, outData.uN, outData.uKnot,
                        u, uBSpline, duBSpline);
                    
                    vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
                    int i = iOffset;
                    for (int b = 0; b < outData.uOrder; b++) // loop from coefficient[uKnot-order] to coefficient[uKnot]
                    {{
                        point.x += uBSpline[b] * texelFetch(uSplineData, i).x;
                        point.y += uBSpline[b] * texelFetch(uSplineData, i+1).x;
                        point.z += uBSpline[b] * texelFetch(uSplineData, i+2).x;
                        i += nDep;
                    }}

                    gl_Position = uProjectionMatrix * point;
                    EmitVertex();
                    u += deltaU;                    
                }}
                EndPrimitive();
            }}
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
            int uKnot, vKnot;
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
            int stride = outData.uN - outData.uOrder + 1;

            outData.uKnot = min(int(mod(gl_InstanceID, stride)) + outData.uOrder, outData.uN);
            outData.vKnot = min(int(gl_InstanceID / stride) + outData.vOrder, outData.vN);
            outData.uFirst = texelFetch(uSplineData, header + outData.uOrder - 1).x; // uKnots[uOrder-1]
            outData.vFirst = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vOrder - 1).x; // vKnots[vOrder-1]
            outData.uSpan = texelFetch(uSplineData, header + outData.uN).x - outData.uFirst; // uKnots[uN] - uKnots[uOrder-1]
            outData.vSpan = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vN).x - outData.vFirst; // vKnots[vN] - vKnots[vOrder-1]
            outData.u = texelFetch(uSplineData, header + outData.uKnot - 1).x; // uKnots[uKnot-1]
            outData.v = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vKnot - 1).x; // vKnots[vKnot-1]
            outData.uInterval = texelFetch(uSplineData, header + outData.uKnot).x - outData.u; // uKnots[uKnot] - uKnots[uKnot-1]
            outData.vInterval = texelFetch(uSplineData, header + outData.uOrder + outData.uN + outData.vKnot).x - outData.v; // vKnots[vKnot] - vKnots[vKnot-1]
            gl_Position = aParameters;
        }
    """

    computeSurfaceSamplesCode = """
        void ComputeSurfaceSamples(in int maxSamples, out float uSamples[3], out float vSamples[3])
        {{
            // Computes sample counts for u and v for the left side ([0]), middle ([1]), and right side ([2]).
            // The left side sample count matches the right side sample count for the previous knot.
            // The middle sample count is the number of samples between knots (same as ComputeCurveSamples).
            float sampleRate[3] = float[3](0.0, 0.0, 0.0);
            if (outData.uInterval > 0.0)
            {{
                float minRate = 1.0 / outData.uInterval;
                if (outData.uOrder < 3)
                {{
                    // It's a plane or point, so just do the minimum sample.
                    sampleRate = float[3](minRate, minRate, minRate);
                }}
                else
                {{
                    float sampleRateLeft[{maxOrder}];
                    float sampleRateRight[{maxOrder}];

                    for (int k = 0; k < outData.uOrder; k++)
                    {{
                        sampleRateLeft[k] = 0.0;
                        sampleRateRight[k] = 0.0;
                    }}
                    for (int j = outData.vKnot-outData.vOrder; j < outData.vKnot; j++)
                    {{
                        int i = max(outData.uKnot - 1 - outData.uOrder, 0);
                        int iLimit = min(outData.uKnot - 1, outData.uN - 2);
                        int coefficientOffset = header + outData.uOrder+outData.uN + outData.vOrder+outData.vN + nDep*outData.uN*j + nDep*i;
                        vec3 coefficient0 = vec3(
                            texelFetch(uSplineData, coefficientOffset+0).x, 
                            texelFetch(uSplineData, coefficientOffset+1).x,
                            texelFetch(uSplineData, coefficientOffset+2).x);
                        coefficientOffset += nDep;
                        vec3 coefficient1 = vec3(
                            texelFetch(uSplineData, coefficientOffset+0).x, 
                            texelFetch(uSplineData, coefficientOffset+1).x,
                            texelFetch(uSplineData, coefficientOffset+2).x);
                        float gap = texelFetch(uSplineData, header + i+outData.uOrder).x - texelFetch(uSplineData, header + i+1).x; // uKnots[i+uOrder] - uKnots[i+1]
                        vec3 dPoint0 = ((outData.uOrder - 1) / gap) * (coefficient1 - coefficient0);
                        while (i < iLimit)
                        {{
                            coefficientOffset += nDep;
                            vec3 coefficient2 = vec3(
                                texelFetch(uSplineData, coefficientOffset+0).x, 
                                texelFetch(uSplineData, coefficientOffset+1).x,
                                texelFetch(uSplineData, coefficientOffset+2).x);
                            gap = texelFetch(uSplineData, header + i+1+outData.uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+1+uOrder] - uKnots[i+2]
                            vec3 dPoint1 = ((outData.uOrder - 1) / gap) * (coefficient2 - coefficient1);
                            gap = texelFetch(uSplineData, header + i+outData.uOrder).x - texelFetch(uSplineData, header + i+2).x; // uKnots[i+uOrder] - uKnots[i+2]
                            vec3 d2Point = ((outData.uOrder - 2) / gap) * (dPoint1 - dPoint0);

                            int k = i - outData.uKnot + 1 + outData.uOrder;
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
            }}
            uSamples[0] = min(floor(0.5 + outData.uInterval * sampleRate[0]), maxSamples);
            uSamples[1] = min(floor(0.5 + outData.uInterval * sampleRate[1]), maxSamples);
            uSamples[2] = min(floor(0.5 + outData.uInterval * sampleRate[2]), maxSamples);

            sampleRate = float[3](0.0, 0.0, 0.0);
            if (outData.vInterval > 0.0)
            {{
                float minRate = 1.0 / outData.vInterval;
                if (outData.vOrder < 3)
                {{
                    // It's a plane or point, so just do the minimum sample.
                    sampleRate = float[3](minRate, minRate, minRate);
                }}
                else
                {{
                    float sampleRateLeft[{maxOrder}];
                    float sampleRateRight[{maxOrder}];

                    for (int k = 0; k < outData.vOrder; k++)
                    {{
                        sampleRateLeft[k] = 0.0;
                        sampleRateRight[k] = 0.0;
                    }}
                    for (int i = outData.uKnot-outData.uOrder; i < outData.uKnot; i++)
                    {{
                        int j = max(outData.vKnot - 1 - outData.vOrder, 0);
                        int jLimit = min(outData.vKnot - 1, outData.vN - 2);
                        int coefficientOffset = header + outData.uOrder+outData.uN + outData.vOrder+outData.vN + nDep*outData.uN*j + nDep*i;
                        vec3 coefficient0 = vec3(
                            texelFetch(uSplineData, coefficientOffset+0).x, 
                            texelFetch(uSplineData, coefficientOffset+1).x,
                            texelFetch(uSplineData, coefficientOffset+2).x);
                        coefficientOffset += nDep*outData.uN;
                        vec3 coefficient1 = vec3(
                            texelFetch(uSplineData, coefficientOffset+0).x, 
                            texelFetch(uSplineData, coefficientOffset+1).x,
                            texelFetch(uSplineData, coefficientOffset+2).x);
                        float gap = texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+outData.vOrder).x - texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+1).x; // vKnots[j+vOrder] - vKnots[j+1]
                        vec3 dPoint0 = ((outData.vOrder - 1) / gap) * (coefficient1 - coefficient0);
                        while (j < jLimit)
                        {{
                            coefficientOffset += nDep*outData.uN;
                            vec3 coefficient2 = vec3(
                                texelFetch(uSplineData, coefficientOffset+0).x, 
                                texelFetch(uSplineData, coefficientOffset+1).x,
                                texelFetch(uSplineData, coefficientOffset+2).x);
                            gap = texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+1+outData.vOrder).x - texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+2).x; // vKnots[j+1+vOrder] - vKnots[j+2]
                            vec3 dPoint1 = ((outData.vOrder - 1) / gap) * (coefficient2 - coefficient1);
                            gap = texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+outData.vOrder).x - texelFetch(uSplineData, header + outData.uOrder+outData.uN + j+2).x; // vKnots[j+vOrder] - vKnots[j+2]
                            vec3 d2Point = ((outData.vOrder - 2) / gap) * (dPoint1 - dPoint0);

                            int k = j - outData.vKnot + 1 + outData.vOrder;
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
            }}
            vSamples[0] = min(floor(0.5 + outData.vInterval * sampleRate[0]), maxSamples);
            vSamples[1] = min(floor(0.5 + outData.vInterval * sampleRate[1]), maxSamples);
            vSamples[2] = min(floor(0.5 + outData.vInterval * sampleRate[2]), maxSamples);
        }}
    """

    surfaceTCShaderCode = """
        #version 410 core

        layout (vertices = 1) out;

        const int header = 4;
        const int nDep = {nDep};

        in SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
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
            int uKnot, vKnot;
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
            ComputeSurfaceSamples(gl_MaxTessGenLevel, uSamples, vSamples);
            gl_TessLevelOuter[0] = vSamples[0] > 0.0 ? vSamples[0] : vSamples[1];
            gl_TessLevelOuter[1] = uSamples[0] > 0.0 ? uSamples[0] : uSamples[1];
            gl_TessLevelOuter[2] = vSamples[2] > 0.0 ? vSamples[2] : vSamples[1];
            gl_TessLevelOuter[3] = uSamples[2] > 0.0 ? uSamples[2] : uSamples[1];
            gl_TessLevelInner[0] = uSamples[1];
            gl_TessLevelInner[1] = vSamples[1];
        }}
    """

    surfaceTEShaderCode = """
        #version 410 core

        layout (quads) in;

        const int header = 4;
        const int nDep = {nDep};

        patch in SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} inData;

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec4 uFillColor;
        uniform samplerBuffer uSplineData;

        flat out SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} outData;
        out vec3 worldPosition;
        out vec3 splineColor;
        out vec3 normal;
        out vec2 parameters;
        out vec2 pixelPer;

        {computeBSplineCode}

        void main()
        {{
            float uBSpline[{maxOrder}];
            float duBSpline[{maxOrder}];
            parameters.x = inData.u + gl_TessCoord.x * inData.uInterval;
            ComputeBSpline(header, inData.uOrder, inData.uN, inData.uKnot, parameters.x, uBSpline, duBSpline);

            float vBSpline[{maxOrder}];
            float dvBSpline[{maxOrder}];
            parameters.y = inData.v + gl_TessCoord.y * inData.vInterval;
            ComputeBSpline(header + inData.uOrder+inData.uN, inData.vOrder, inData.vN, inData.vKnot, parameters.y, vBSpline, dvBSpline);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
            vec3 duPoint = vec3(0.0, 0.0, 0.0);
            vec3 dvPoint = vec3(0.0, 0.0, 0.0);
            {initializeSplineColor}
            int j = header + inData.uOrder+inData.uN + inData.vOrder+inData.vN + (inData.vKnot - inData.vOrder) * inData.uN * nDep;
            for (int vB = 0; vB < inData.vOrder; vB++)
            {{
                int i = j + (inData.uKnot - inData.uOrder) * nDep;
                for (int uB = 0; uB < inData.uOrder; uB++)
                {{
                    point.x += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                    point.y += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    point.z += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    duPoint.x += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                    duPoint.y += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    duPoint.z += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    dvPoint.x += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i).x;
                    dvPoint.y += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+1).x;
                    dvPoint.z += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+2).x;
                    {computeSplineColor}
                    i += nDep;
                }}
                j += inData.uN * nDep;
            }}
            {postProcessSplineColor}

            outData = inData;

            worldPosition = point.xyz;
            normal = normalize(cross(duPoint, dvPoint));
            float zScale = 1.0 / (point.z * point.z);
            pixelPer.x = zScale * max(uScreenScale.x * abs(point.x * duPoint.z - duPoint.x * point.z), uScreenScale.y * abs(point.y * duPoint.z - duPoint.y * point.z));
            pixelPer.y = zScale * max(uScreenScale.x * abs(point.x * dvPoint.z - dvPoint.x * point.z), uScreenScale.y * abs(point.y * dvPoint.z - dvPoint.y * point.z));
            gl_Position = uProjectionMatrix * point;
        }}
    """

    surfaceGeometryShaderCode = """
        #version 330 core

        layout( points ) in;
        layout( triangle_strip, max_vertices = 256 ) out;

        const int header = 4;
        const int nDep = {nDep};

        in SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} inData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform vec4 uFillColor;
        uniform vec3 uLightDirection;
        uniform samplerBuffer uSplineData;

        out vec3 splineColor; // We restrict our output to color to reduce the number of components per vertex.

        struct SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }} outData; // We don't output outData (too many components per vertex), but we do use it in ComputeSurfaceSamples.

        {computeSampleRateCode}

        {computeSurfaceSamplesCode}

        {computeBSplineCode}

        void main() 
        {{
            float uFullSamples[3];
            float vFullSamples[3];

            outData.uOrder = inData[0].uOrder;
            outData.vOrder = inData[0].vOrder;
            outData.uN = inData[0].uN;
            outData.vN = inData[0].vN;
            outData.uKnot = inData[0].uKnot;
            outData.vKnot = inData[0].vKnot;
            outData.u = inData[0].u;
            outData.v = inData[0].v;
            outData.uInterval = inData[0].uInterval;
            outData.vInterval = inData[0].vInterval;
            int maxVertices = gl_MaxGeometryTotalOutputComponents / 7; // The number of output components per vertex is 7 = position.xyzw + splineColor.rgb
            ComputeSurfaceSamples(maxVertices, uFullSamples, vFullSamples);

            if (uFullSamples[1] > 0.0 && vFullSamples[1] > 0.0)
            {{
                float alpha = maxVertices / (2.0 * uFullSamples[1] * (vFullSamples[1] + 1.0));
                if (alpha < 1.0)
                {{
                    alpha = sqrt(alpha);
                    uFullSamples[1] = alpha * uFullSamples[1];
                    vFullSamples[1] = alpha * (vFullSamples[1] + 1.0) - 1.0;
                }}
                int uSamples = int(uFullSamples[1]);
                int vSamples = int(vFullSamples[1]);

                float uBSpline[{maxOrder}];
                float duBSpline[{maxOrder}];
                float uBSplineNext[{maxOrder}];
                float duBSplineNext[{maxOrder}];
                float vBSpline[{maxOrder}];
                float dvBSpline[{maxOrder}];
                float deltaU = outData.uInterval / uSamples;
                float deltaV = outData.vInterval / vSamples;
                float u = outData.u;
                ComputeBSpline(header, outData.uOrder, outData.uN, outData.uKnot, u, uBSpline, duBSpline);

                int jOffset = header + outData.uOrder+outData.uN + outData.vOrder+outData.vN + (outData.vKnot - outData.vOrder) * outData.uN * nDep;
                int iOffset = (outData.uKnot - outData.uOrder) * nDep;

                for (int uSample = 0; uSample < uSamples; uSample++)
                {{
                    float uNext = u + deltaU;
                    ComputeBSpline(header, outData.uOrder, outData.uN, outData.uKnot, uNext, uBSplineNext, duBSplineNext);
                        
                    float v = outData.v;
                    for (int vSample = 0; vSample <= vSamples; vSample++)
                    {{
                        ComputeBSpline(header + outData.uOrder+outData.uN, outData.vOrder, outData.vN, outData.vKnot, v, vBSpline, dvBSpline);

                        vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
                        vec3 duPoint = vec3(0.0, 0.0, 0.0);
                        vec3 dvPoint = vec3(0.0, 0.0, 0.0);
                        {initializeSplineColor}
                        int j = jOffset;
                        for (int vB = 0; vB < outData.vOrder; vB++)
                        {{
                            int i = j + iOffset;
                            for (int uB = 0; uB < outData.uOrder; uB++)
                            {{
                                point.x += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                                point.y += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                                point.z += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                                duPoint.x += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                                duPoint.y += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                                duPoint.z += duBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                                dvPoint.x += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i).x;
                                dvPoint.y += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+1).x;
                                dvPoint.z += uBSpline[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+2).x;
                                {computeSplineColor}
                                i += nDep;
                            }}
                            j += outData.uN * nDep;
                        }}
                        {postProcessSplineColor}
                        vec3 normal = normalize(cross(duPoint, dvPoint));
                        float specular = pow(abs(dot(normal, normalize(uLightDirection + point.xyz / length(point)))), 25.0);
                        splineColor = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * splineColor;
                        gl_Position = uProjectionMatrix * point;
                        EmitVertex();

                        point = vec4(0.0, 0.0, 0.0, 1.0);
                        duPoint = vec3(0.0, 0.0, 0.0);
                        dvPoint = vec3(0.0, 0.0, 0.0);
                        {initializeSplineColor}
                        j = jOffset;
                        for (int vB = 0; vB < outData.vOrder; vB++)
                        {{
                            int i = j + iOffset;
                            for (int uB = 0; uB < outData.uOrder; uB++)
                            {{
                                point.x += uBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                                point.y += uBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                                point.z += uBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                                duPoint.x += duBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i).x;
                                duPoint.y += duBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+1).x;
                                duPoint.z += duBSplineNext[uB] * vBSpline[vB] * texelFetch(uSplineData, i+2).x;
                                dvPoint.x += uBSplineNext[uB] * dvBSpline[vB] * texelFetch(uSplineData, i).x;
                                dvPoint.y += uBSplineNext[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+1).x;
                                dvPoint.z += uBSplineNext[uB] * dvBSpline[vB] * texelFetch(uSplineData, i+2).x;
                                {computeSplineColor}
                                i += nDep;
                            }}
                            j += outData.uN * nDep;
                        }}
                        {postProcessSplineColor}
                        normal = normalize(cross(duPoint, dvPoint));
                        specular = pow(abs(dot(normal, normalize(uLightDirection + point.xyz / length(point)))), 25.0);
                        splineColor = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * splineColor;
                        gl_Position = uProjectionMatrix * point;
                        EmitVertex();

                        v += deltaV;                    
                    }}
                    EndPrimitive();
                    u = uNext;
                    uBSpline = uBSplineNext;
                    duBSpline = duBSplineNext;
                }}
            }}
        }}
    """

    surfaceSimpleFragmentShaderCode = """
        #version 330 core
     
        in vec3 splineColor;
        uniform vec4 uFillColor;
        out vec4 color;
     
        void main() {
            color = vec4(splineColor, uFillColor.a);
        }
    """

    surfaceFragmentShaderCode = """
        #version 410 core
     
        flat in SplineInfo
        {
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        } inData;
        in vec3 worldPosition;
        in vec3 splineColor;
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
            float specular = pow(abs(dot(normal, normalize(uLightDirection + worldPosition / length(worldPosition)))), 25.0);
            bool line = (uOptions & (1 << 2)) > 0 && (pixelPer.x * (parameters.x - inData.uFirst) < 1.5 || pixelPer.x * (inData.uFirst + inData.uSpan - parameters.x) < 1.5);
            line = line || ((uOptions & (1 << 2)) > 0 && (pixelPer.y * (parameters.y - inData.vFirst) < 1.5 || pixelPer.y * (inData.vFirst + inData.vSpan - parameters.y) < 1.5));
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.x * (parameters.x - inData.u) < 1.5);
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.y * (parameters.y - inData.v) < 1.5);
            color = line ? uLineColor : ((uOptions & (1 << 1)) > 0 ? vec4(splineColor, uFillColor.a) : vec4(0.0, 0.0, 0.0, 0.0));
            color.rgb = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * color.rgb;
            if (color.a == 0.0)
                discard;
        }
    """
 
    def __init__(self, *args, eye=(0.0, 0.0, 3.0), center=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), **kw):
        OpenGLFrame.__init__(self, *args, **kw)
        self.animate = 0 # Set to number of milliseconds before showing next frame (0 means no animation)

        self.splineDrawList = []
        
        self.origin = None
        self.button = None
        self.mode = self.ROTATE
        self.speed = 0.01

        self.backgroundColor = np.array((0.0, 0.2, 0.2, 1.0), np.float32)

        self.SetDefaultView(eye, center, up)
        self.ResetView()

        self.bind("<ButtonPress>", self.MouseDown)
        self.bind("<Motion>", self.MouseMove)
        self.bind("<ButtonRelease>", self.MouseUp)
        self.bind("<MouseWheel>", self.MouseWheel)
        self.bind("<Unmap>", self.Unmap)

        self.tessellationEnabled = True
        self.glInitialized = False

    def SetDefaultView(self, eye, center, up):
        """
        Set the default view values used when resetting the view.
        """
        self.defaultEye = np.array(eye, np.float32)
        self.defaultCenter = np.array(center, np.float32)
        self.defaultUp = np.array(up, np.float32)
        self.defaultUp = self.defaultUp / np.linalg.norm(self.defaultUp)
    
    def SetBackgroundColor(self, r, g=None, b=None, a=None):
        """
        Set the background color for the frame.

        Parameters
        ----------
        r : `float`, `int` or array-like of floats or ints
            The red value [0, 1] as a float, [0, 255] as an int, or the rgb or rgba value as floats or ints (default).
        
        g: `float` or `int`
            The green value [0, 1] as a float or [0, 255] as an int.
        
        b: `float` or `int`
            The blue value [0, 1] as a float or [0, 255] as an int.
        
        a: `float`, `int`, or None
            The alpha value [0, 1] as a float or [0, 255] as an int. If `None` then alpha is set to 1.
        """
        self.backgroundColor = _set_color(r, g, b, a)
        if self.glInitialized:
            glClearColor(self.backgroundColor[0], self.backgroundColor[1], self.backgroundColor[2], self.backgroundColor[3])

    def SetSplineList(self, list):
        """
        Set the `DrawableSpline` list which determines the splines to display.
        """
        self.splineDrawList = list
    
    def ResetView(self):
        """
        Update the view position to default values.
        """
        self.eye = self.defaultEye.copy()
        self.look = self.defaultEye - self.defaultCenter
        self.anchorDistance = np.linalg.norm(self.look)
        self.look = self.look / self.anchorDistance
        self.up = self.defaultUp.copy()
        self.horizon = np.cross(self.up, self.look)
        self.horizon = self.horizon / np.linalg.norm(self.horizon)
        self.vertical = np.cross(self.look, self.horizon)
        self.anchorPosition = self.eye - self.anchorDistance * self.look

    def initgl(self):
        """
        Handle `OpenGLFrame` initgl action. Calls `CreateGLResources` and `HandleScreenSizeUpdate`.
        """
        if not self.glInitialized:
            self.CreateGLResources()
            self.glInitialized = True

        self.ResetBounds()

    def CreateGLResources(self):
        """
        Create OpenGL resources upon creation of the frame and window recovery (un-minimize).
        """
        #print("GL_VERSION: ", glGetString(GL_VERSION))
        #print("GL_SHADING_LANGUAGE_VERSION: ", glGetString(GL_SHADING_LANGUAGE_VERSION))
        #print("GL_MAX_TESS_GEN_LEVEL: ", glGetIntegerv(GL_MAX_TESS_GEN_LEVEL))

        # Set up GL texture buffer for spline data
        self.splineDataBuffer = glGenBuffers(1)
        self.splineTextureBuffer = glGenTextures(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.splineDataBuffer)
        glBindTexture(GL_TEXTURE_BUFFER, self.splineTextureBuffer)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.splineDataBuffer)
        glBufferData(GL_TEXTURE_BUFFER, 4 * DrawableSpline._maxFloats, None, GL_STATIC_READ)

        # Set light direction
        self.lightDirection = np.array((0.63960218, 0.63960218, 0.42640144), np.float32)
        self.lightDirection = self.lightDirection / np.linalg.norm(self.lightDirection)

        # Bind parameter buffer
        self.parameterBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.parameterBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * 4, np.array([0,0,0,0], np.float32), GL_STATIC_DRAW)

        # Compile shaders and link programs
        self.computeBSplineCode = self.computeBSplineCode.format(maxOrder=DrawableSpline.maxOrder)
        self.computeSurfaceSamplesCode = self.computeSurfaceSamplesCode.format(maxOrder=DrawableSpline.maxOrder)
        self.tessellationEnabled = True
        try:
            # Must create CurveProgram first, because it checks and potentially resets tessellationEnabled flag.
            self.curveProgram = CurveProgram(self)
            self.surface3Program = SurfaceProgram(self, 3, "", "", "splineColor = uFillColor.rgb;")
            self.surface4Program = SurfaceProgram(self, 4, "splineColor = vec3(0.0, 0.0, 0.0);",
                """
                    splineColor.r += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+3).x;
                """,
                # Taken from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
                # uFillColor is passed in as HSV
                """
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(uFillColor.xxx + K.xyz) * 6.0 - K.www);
                    splineColor = uFillColor.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), splineColor.r);
                """)
            self.surface6Program = SurfaceProgram(self, 6, "splineColor = vec3(0.0, 0.0, 0.0);",
                """
                    splineColor.r += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+3).x;
                    splineColor.g += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+4).x;
                    splineColor.b += uBSpline[uB] * vBSpline[vB] * texelFetch(uSplineData, i+5).x;
                """, "")
        except shaders.ShaderCompilationError as exception:
            error = exception.args[0]
            lineNumber = error.split(":")[3]
            source = exception.args[1][0]
            badLine = source.split(b"\n")[int(lineNumber)-1]
            shaderType = exception.args[2]
            print(shaderType, error)
            print(badLine)
            quit()

        glUseProgram(0)
        glEnable( GL_DEPTH_TEST )
        glClearColor(self.backgroundColor[0], self.backgroundColor[1], self.backgroundColor[2], self.backgroundColor[3])

    def ResetBounds(self):
        """
        Handle window size and/or clipping plane update (typically after a window resize).
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        xExtent = self.width / self.height
        defaultAnchorDistance = np.linalg.norm(self.defaultEye - self.defaultCenter)
        clipDistance = defaultAnchorDistance / np.sqrt(3.0)
        near = 0.01 * defaultAnchorDistance / 3.0
        far = defaultAnchorDistance + clipDistance
        top = clipDistance * near / defaultAnchorDistance # Choose frustum that displays [-clipDistance,clipDistance] in y for z = -defaultAnchorDistance
        glFrustum(-top*xExtent, top*xExtent, -top, top, near, far)
        #glOrtho(-xExtent, xExtent, -1.0, 1.0, -1.0, 1.0)

        self.projection = glGetFloatv(GL_PROJECTION_MATRIX)
        self.screenScale = np.array((0.5 * self.height * self.projection[0,0], 0.5 * self.height * self.projection[1,1], 1.0), np.float32)
        self.clipBounds = np.array((1.0 / self.projection[0,0], 1.0 / self.projection[1,1], -far, -near), np.float32)

        self.curveProgram.ResetBounds(self)
        self.surface3Program.ResetBounds(self)
        self.surface4Program.ResetBounds(self)
        self.surface6Program.ResetBounds(self)

        glUseProgram(0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def redraw(self):
        """
        Handle `OpenGLFrame` redraw action. Updates view and draws spline list.
        """
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glLoadIdentity()

        if self.button is not None:
            if (self.mode == self.ROTATE and self.button == 1) or (self.mode == self.PAN and self.button == 3):
                ratio = self.anchorDistance / (2 * 0.4142 * self.height)
                self.eye = self.eye - ((self.current[0] - self.origin[0]) * ratio) * self.horizon + \
                    ((self.current[1] - self.origin[1]) * ratio) * self.vertical
                self.look = self.eye - self.anchorPosition
                self.look = self.look / np.linalg.norm(self.look)
                self.eye = self.anchorPosition + self.anchorDistance * self.look
                self.origin = self.current
            elif (self.mode == self.PAN and self.button == 1) or (self.mode == self.ROTATE and self.button == 3):
                ratio = self.anchorDistance / (2 * 0.4142 * self.height)
                self.eye = self.eye - ((self.current[0] - self.origin[0]) * ratio) * self.horizon + \
                    ((self.current[1] - self.origin[1]) * ratio) * self.vertical
                self.anchorPosition = self.eye - self.anchorDistance * self.look
                self.origin = self.current
            elif self.mode == self.FLY:
                self.vertical = self.vertical + 0.5 * self.up
                self.vertical = self.vertical / np.linalg.norm(self.vertical)
                x = (1.2 * 50 / 1000) * (self.width - 2 * self.current[0]) / self.width
                y = (-1.2 * 50 / 1000) * (self.height - 2 * self.current[1]) / self.height
                self.look = self.look + x * self.horizon + y * self.vertical
                self.look = self.look / np.linalg.norm(self.look)
                if self.button == 1: # Left button
                    self.eye = self.eye - self.speed * self.look
                elif self.button == 2: # Wheel button
                    self.eye = self.eye + self.speed * self.look
                self.anchorPosition = self.eye - self.anchorDistance * self.look

        self.horizon = np.cross(self.vertical, self.look)
        self.horizon = self.horizon / np.linalg.norm(self.horizon)
        self.vertical = np.cross(self.look, self.horizon)
        transform = np.array(
            ((self.horizon[0], self.vertical[0], self.look[0], 0.0),
            (self.horizon[1], self.vertical[1], self.look[1], 0.0),
            (self.horizon[2], self.vertical[2], self.look[2], 0.0),
            (-np.dot(self.horizon, self.eye), -np.dot(self.vertical, self.eye), -np.dot(self.look, self.eye), 1.0)), np.float32)

        for spline in self.splineDrawList:
            spline._Draw(self, transform)

        glFlush()

    def Unmap(self, event):
        """
        Handle window unmap.
        """
        self.glInitialized = False

    def Update(self):
        """
        Update the frame, typically after updating the spline list.
        """
        self.tkExpose(None)

    def Reset(self):
        """
        Reset the view and update the frame.
        """
        self.ResetView()
        self.Update()
    
    def SetMode(self, mode):
        """
        Set the view mode for the frame.

        Parameters
        ----------
        mode : `int` with the following values:
            * `SplineOpenGLFrame.ROTATE` Dragging the left mouse rotates the view.
            * `SplineOpenGLFrame.PAN` Dragging the left mouse pans the view.
            * `SplineOpenGLFrame.FLY` Dragging the left mouse flies toward the mouse position.
        """
        self.mode = mode
    
    def SetScale(self, scale):
        """
        Set the flying speed scale.

        Parameters
        ----------
        scale : `float`
            Speed scale between 0 and 1.
        """
        self.speed = 0.1 * (100.0 ** float(scale) - 1.0) / 99.0

    def MouseDown(self, event):
        """
        Handle mouse down event.
        """
        self.origin = np.array((event.x, event.y), np.float32)
        self.current = self.origin
        self.button = event.num

        if self.button == 4 or self.button == 5: # MouseWheel
            self.anchorDistance *= 0.9 if self.button == 4 else 1.1
            self.eye = self.anchorPosition + self.anchorDistance * self.look
            self.Update()
        
        if self.mode == self.FLY:
            self.animate = 50 # Update every 20th of a second
            self.Update()

    def MouseMove(self, event):
        """
        Handle mouse move event.
        """
        self.current = np.array((event.x, event.y), np.float32)
        if self.mode == self.ROTATE or self.mode == self.PAN:
            self.Update()

    def MouseUp(self, event):
        """
        Handle mouse up event.
        """
        self.origin = None
        self.button = None
        if self.mode == self.FLY:
            self.animate = 0 # Stop animation
            self.Update()

    def MouseWheel(self, event):
        """
        Handle mouse wheel event.
        """
        if event.delta < 0:
            self.anchorDistance *= 1.1
        elif event.delta > 0:
            self.anchorDistance *= 0.9
        self.eye = self.anchorPosition + self.anchorDistance * self.look
        self.Update()

class CurveProgram:
    """ Compile curve program """
    def __init__(self, frame):
        try:
            curveTCShader = shaders.compileShader(frame.curveTCShaderCode.format(
                computeSampleRateCode=frame.computeSampleRateCode,
                computeCurveSamplesCode=frame.computeCurveSamplesCode), GL_TESS_CONTROL_SHADER)
        except shaders.ShaderCompilationError:
            frame.tessellationEnabled = False

        if frame.tessellationEnabled:
            self.curveProgram = shaders.compileProgram(
                shaders.compileShader(frame.curveVertexShaderCode, GL_VERTEX_SHADER), 
                curveTCShader, 
                shaders.compileShader(frame.curveTEShaderCode.format(
                    computeBSplineCode=frame.computeBSplineCode,
                    maxOrder=DrawableSpline.maxOrder), GL_TESS_EVALUATION_SHADER), 
                shaders.compileShader(frame.curveFragmentShaderCode, GL_FRAGMENT_SHADER))
        else:
            self.curveProgram = shaders.compileProgram(
                shaders.compileShader(frame.curveVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(frame.curveGeometryShaderCode.format(
                    computeSampleRateCode=frame.computeSampleRateCode,
                    computeCurveSamplesCode=frame.computeCurveSamplesCode,
                    computeBSplineCode=frame.computeBSplineCode,
                    maxOrder=DrawableSpline.maxOrder), GL_GEOMETRY_SHADER), 
                shaders.compileShader(frame.curveFragmentShaderCode, GL_FRAGMENT_SHADER))

        glUseProgram(self.curveProgram)
        self.aCurveParameters = glGetAttribLocation(self.curveProgram, "aParameters")
        glBindBuffer(GL_ARRAY_BUFFER, frame.parameterBuffer)
        glVertexAttribPointer(self.aCurveParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
        self.uCurveProjectionMatrix = glGetUniformLocation(self.curveProgram, 'uProjectionMatrix')
        self.uCurveScreenScale = glGetUniformLocation(self.curveProgram, 'uScreenScale')
        self.uCurveClipBounds = glGetUniformLocation(self.curveProgram, 'uClipBounds')
        self.uCurveLineColor = glGetUniformLocation(self.curveProgram, 'uLineColor')
        self.uCurveSplineData = glGetUniformLocation(self.curveProgram, 'uSplineData')
        glUniform1i(self.uCurveSplineData, 0) # GL_TEXTURE0 is the spline buffer texture
    
    def ResetBounds(self, frame):
        """Reset bounds and other frame configuration for curve program"""
        glUseProgram(self.curveProgram)
        glUniformMatrix4fv(self.uCurveProjectionMatrix, 1, GL_FALSE, frame.projection)
        glUniform3fv(self.uCurveScreenScale, 1, frame.screenScale)
        glUniform4fv(self.uCurveClipBounds, 1, frame.clipBounds)

class SurfaceProgram:
    """ Compile surface program """
    def __init__(self, frame, nDep, initializeSplineColor, computeSplineColor, postProcessSplineColor):
        if frame.tessellationEnabled:
            self.surfaceProgram = shaders.compileProgram(
                shaders.compileShader(frame.surfaceVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(frame.surfaceTCShaderCode.format(
                    nDep=nDep,
                    computeSampleRateCode=frame.computeSampleRateCode,
                    computeSurfaceSamplesCode=frame.computeSurfaceSamplesCode), GL_TESS_CONTROL_SHADER), 
                shaders.compileShader(frame.surfaceTEShaderCode.format(
                    nDep=nDep,
                    computeBSplineCode=frame.computeBSplineCode,
                    initializeSplineColor=initializeSplineColor,
                    computeSplineColor=computeSplineColor,
                    postProcessSplineColor=postProcessSplineColor,
                    maxOrder=DrawableSpline.maxOrder), GL_TESS_EVALUATION_SHADER), 
                shaders.compileShader(frame.surfaceFragmentShaderCode, GL_FRAGMENT_SHADER))
        else:
            self.surfaceProgram = shaders.compileProgram(
                shaders.compileShader(frame.surfaceVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(frame.surfaceGeometryShaderCode.format(
                    nDep=nDep,
                    computeSampleRateCode=frame.computeSampleRateCode,
                    computeSurfaceSamplesCode=frame.computeSurfaceSamplesCode,
                    computeBSplineCode=frame.computeBSplineCode,
                    initializeSplineColor=initializeSplineColor,
                    computeSplineColor=computeSplineColor,
                    postProcessSplineColor=postProcessSplineColor,
                    maxOrder=DrawableSpline.maxOrder), GL_GEOMETRY_SHADER), 
                shaders.compileShader(frame.surfaceSimpleFragmentShaderCode, GL_FRAGMENT_SHADER))

        glUseProgram(self.surfaceProgram)
        self.aSurfaceParameters = glGetAttribLocation(self.surfaceProgram, "aParameters")
        glBindBuffer(GL_ARRAY_BUFFER, frame.parameterBuffer)
        glVertexAttribPointer(self.aSurfaceParameters, 4, GL_FLOAT, GL_FALSE, 0, None)
        self.uSurfaceProjectionMatrix = glGetUniformLocation(self.surfaceProgram, 'uProjectionMatrix')
        self.uSurfaceScreenScale = glGetUniformLocation(self.surfaceProgram, 'uScreenScale')
        self.uSurfaceClipBounds = glGetUniformLocation(self.surfaceProgram, 'uClipBounds')
        self.uSurfaceFillColor = glGetUniformLocation(self.surfaceProgram, 'uFillColor')
        self.uSurfaceLineColor = glGetUniformLocation(self.surfaceProgram, 'uLineColor')
        glUniform3fv(glGetUniformLocation(self.surfaceProgram, 'uLightDirection'), 1, frame.lightDirection)
        self.uSurfaceOptions = glGetUniformLocation(self.surfaceProgram, 'uOptions')
        self.uSurfaceSplineData = glGetUniformLocation(self.surfaceProgram, 'uSplineData')
        glUniform1i(self.uSurfaceSplineData, 0) # GL_TEXTURE0 is the spline buffer texture
    
    def ResetBounds(self, frame):
        """Reset bounds and other frame configuration for surface program"""
        glUseProgram(self.surfaceProgram)
        glUniformMatrix4fv(self.uSurfaceProjectionMatrix, 1, GL_FALSE, frame.projection)
        glUniform3fv(self.uSurfaceScreenScale, 1, frame.screenScale)
        glUniform4fv(self.uSurfaceClipBounds, 1, frame.clipBounds)
