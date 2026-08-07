from collections import namedtuple
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from bspy.manifold import Manifold
import OpenGL.GL.shaders as shaders
try:
    from pyopengltk import OpenGLFrame
except ImportError:
    from tkinter import Frame as OpenGLFrame
from bspy import Spline

class SplineOpenGLFrame(OpenGLFrame):
    """
    A tkinter `OpenGLFrame` with shaders to display a `Spline`.
    """

    ROTATE = 1
    """Default view mode in which dragging the left mouse rotates the view."""
    PAN = 2
    """View mode in which dragging the left mouse pans the view."""
    FLY = 3
    """View mode in which dragging the left mouse flies toward the mouse position."""

    MsPerFrame = 50 # Update every 20th of a second
    """Milliseconds per frame when animating or flying."""

    maxOrder = 9
    """Maximum order for drawable splines."""
    maxKnots = 1024
    """Maximum number of 2D knots for drawable splines (order[0] + nCoef[0] + order[1] + nCoef[1] + 4, includes 4 header values)."""
    maxCoefficients = 16384
    """Maximum number of 2D coefficients for drawable splines (nCoef[0] * nCoef[1], textures must support this width)."""

    HULL = (1 << 0)
    """Option to draw the convex hull of the spline (the coefficients). Off by default."""
    SHADED = (1 << 1)
    """Option to draw the spline shaded (only useful for nInd >= 2). On by default."""
    BOUNDARY = (1 << 2)
    """Option to draw the boundary of the spline in the line color (only useful for nInd >= 2). On by default."""
    ISOPARMS = (1 << 3)
    """Option to draw the lines of constant knot values of the spline in the line color (only useful for nInd >= 2). Off by default."""

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
                    float knotValue = texelFetch(uKnots, offset + i).x; // knots[i]
                    float alpha = (u - knotValue) / (texelFetch(uKnots, offset + i + degree).x - knotValue); // (u - knots[i]) / (knots[i+degree] - knots[i]);
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
                    float knotValue = texelFetch(uKnots, offset + i).x; // knots[i]
                    float gap = texelFetch(uKnots, offset + i + degree).x - knotValue; // knots[i+degree] - knots[i]
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
        #version 330 core
     
        const int header = 2;

        uniform samplerBuffer uKnots;

        struct SplineInfo
        {
            int uOrder;
            int uN;
            int uKnot;
            float u;
            float uInterval;
        };
        out SplineInfo vertexData;

        void main()
        {
            vertexData.uOrder = int(texelFetch(uKnots, 0).x);
            vertexData.uN = int(texelFetch(uKnots, 1).x);
            vertexData.uKnot = min(gl_InstanceID + vertexData.uOrder, vertexData.uN);
            vertexData.u = texelFetch(uKnots, header + vertexData.uKnot - 1).x; // knots[uKnot-1]
            vertexData.uInterval = texelFetch(uKnots, header + vertexData.uKnot).x - vertexData.u; // knots[uKnot] - knots[uKnot-1]
        }
    """

    computeCurveSamplesCode = """
        void ComputeCurveSamples(in int maxSamples, inout SplineInfo samplesData, out float uSamples)
        {
            float sampleRate = 0.0;
            if (samplesData.uInterval > 0.0)
            {
                float minRate = 1.0 / samplesData.uInterval;
                if (samplesData.uOrder < 3)
                {
                    // It's a line or point, so just do the minimum sample.
                    sampleRate = minRate;
                }
                else
                {
                    int i = samplesData.uKnot - samplesData.uOrder;
                    int coefficientOffset = i;
                    vec3 coefficient0 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                    coefficientOffset++;
                    vec3 coefficient1 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                    float gap = texelFetch(uKnots, header + i+samplesData.uOrder).x - texelFetch(uKnots, header + i+1).x; // uKnots[i+uOrder] - uKnots[i+1]
                    vec3 dPoint0 = ((samplesData.uOrder - 1) / gap) * (coefficient1 - coefficient0);
                    while (i < samplesData.uKnot-2)
                    {
                        coefficientOffset++;
                        vec3 coefficient2 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                        gap = texelFetch(uKnots, header + i+1+samplesData.uOrder).x - texelFetch(uKnots, header + i+2).x; // uKnots[i+1+uOrder] - uKnots[i+2]
                        vec3 dPoint1 = ((samplesData.uOrder - 1) / gap) * (coefficient2 - coefficient1);
                        gap = texelFetch(uKnots, header + i+samplesData.uOrder).x - texelFetch(uKnots, header + i+2).x; // uKnots[i+uOrder] - uKnots[i+2]
                        vec3 d2Point = ((samplesData.uOrder - 2) / gap) * (dPoint1 - dPoint0);

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
            uSamples = min(floor(0.5 + samplesData.uInterval * sampleRate), maxSamples);
        }
    """

    curveTCShaderCode = """
        #version 410 core

        layout (vertices = 1) out;

        const int header = 2;

        struct SplineInfo
        {{
            int uOrder;
            int uN;
            int uKnot;
            float u;
            float uInterval;
        }};
        in SplineInfo vertexData[];

        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform samplerBuffer uKnots;
        uniform samplerBuffer uXYZCoefs;

        patch out SplineInfo tcData;

        {computeSampleRateCode}

        {computeCurveSamplesCode}

        void main()
        {{
            tcData = vertexData[gl_InvocationID];

            float uSamples = 0.0;
            ComputeCurveSamples(gl_MaxTessGenLevel, tcData, uSamples);
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = uSamples;
            gl_TessLevelOuter[1] = 10.0;
        }}
    """

    curveTEShaderCode = """
        #version 410 core

        layout (isolines) in;

        const int header = 2;

        struct SplineInfo
        {{
            int uOrder;
            int uN;
            int uKnot;
            float u;
            float uInterval;
        }};
        patch in SplineInfo tcData;

        uniform mat4 uProjectionMatrix;
        uniform samplerBuffer uKnots;
        uniform samplerBuffer uXYZCoefs;

        {computeBSplineCode}

        void main()
        {{
            float uBSpline[{maxOrder}];
            float duBSpline[{maxOrder}];
            ComputeBSpline(header, tcData.uOrder, tcData.uN, tcData.uKnot,
                tcData.u + gl_TessCoord.x * tcData.uInterval, 
                uBSpline, duBSpline);
            
            vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
            int i = tcData.uKnot - tcData.uOrder;
            for (int b = 0; b < tcData.uOrder; b++) // loop from coefficient[uKnot-order] to coefficient[uKnot]
            {{
                point.xyz += uBSpline[b] * texelFetch(uXYZCoefs, i).xyz;
                i++;
            }}

            gl_Position = uProjectionMatrix * point;
            //gl_Position = vec4(0.5 * gl_TessCoord.x, tcData.u, 0.0, 1.0);
        }}
    """

    curveGeometryShaderCode = """
        #version 330 core

        layout( points ) in;
        layout( line_strip, max_vertices = 256 ) out;

        const int header = 2;

        struct SplineInfo
        {{
            int uOrder;
            int uN;
            int uKnot;
            float u;
            float uInterval;
        }};
        in SplineInfo vertexData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform samplerBuffer uKnots;
        uniform samplerBuffer uXYZCoefs;

        SplineInfo geometryData; // We don't output geometryData (too many components per vertex), but we do use it in ComputeCurveSamples.

        {computeSampleRateCode}

        {computeCurveSamplesCode}

        {computeBSplineCode}

        void main()
        {{
            float uSamples = 0.0;

            geometryData = vertexData[0];
            ComputeCurveSamples(gl_MaxGeometryOutputVertices - 1, geometryData, uSamples);

            if (uSamples > 0.0)
            {{
                float uBSpline[{maxOrder}];
                float duBSpline[{maxOrder}];
                float u = geometryData.u;
                float deltaU = geometryData.uInterval / uSamples;
                int iOffset = geometryData.uKnot - geometryData.uOrder;

                for (int uSample = 0; uSample <= uSamples; uSample++)
                {{
                    ComputeBSpline(header, geometryData.uOrder, geometryData.uN, geometryData.uKnot,
                        u, uBSpline, duBSpline);
                    
                    vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
                    int i = iOffset;
                    for (int b = 0; b < geometryData.uOrder; b++) // loop from coefficient[uKnot-order] to coefficient[uKnot]
                    {{
                        point.xyz += uBSpline[b] * texelFetch(uXYZCoefs, i).xyz;
                        i++;
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
        #version 330 core

        const int header = 4;

        uniform samplerBuffer uKnots;

        struct SplineInfo
        {
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        };
        out SplineInfo vertexData;

        void main()
        {
            vertexData.uOrder = int(texelFetch(uKnots, 0).x);
            vertexData.vOrder = int(texelFetch(uKnots, 1).x);
            vertexData.uN = int(texelFetch(uKnots, 2).x);
            vertexData.vN = int(texelFetch(uKnots, 3).x);
            int stride = vertexData.uN - vertexData.uOrder + 1;
            int strides = gl_InstanceID / stride;

            vertexData.uKnot = gl_InstanceID - stride * strides + vertexData.uOrder;
            vertexData.vKnot = strides + vertexData.vOrder;
            vertexData.uFirst = texelFetch(uKnots, header + vertexData.uOrder - 1).x; // uKnots[uOrder-1]
            vertexData.vFirst = texelFetch(uKnots, header + vertexData.uOrder + vertexData.uN + vertexData.vOrder - 1).x; // vKnots[vOrder-1]
            vertexData.uSpan = texelFetch(uKnots, header + vertexData.uN).x - vertexData.uFirst; // uKnots[uN] - uKnots[uOrder-1]
            vertexData.vSpan = texelFetch(uKnots, header + vertexData.uOrder + vertexData.uN + vertexData.vN).x - vertexData.vFirst; // vKnots[vN] - vKnots[vOrder-1]
            vertexData.u = texelFetch(uKnots, header + vertexData.uKnot - 1).x; // uKnots[uKnot-1]
            vertexData.v = texelFetch(uKnots, header + vertexData.uOrder + vertexData.uN + vertexData.vKnot - 1).x; // vKnots[vKnot-1]
            vertexData.uInterval = texelFetch(uKnots, header + vertexData.uKnot).x - vertexData.u; // uKnots[uKnot] - uKnots[uKnot-1]
            vertexData.vInterval = texelFetch(uKnots, header + vertexData.uOrder + vertexData.uN + vertexData.vKnot).x - vertexData.v; // vKnots[vKnot] - vKnots[vKnot-1]
        }
    """

    computeSurfaceSamplesCode = """
        void ComputeSurfaceSamples(in int maxSamples, inout SplineInfo samplesData, out float uSamples[3], out float vSamples[3])
        {{
            // Computes sample counts for u and v for the left side ([0]), middle ([1]), and right side ([2]).
            // The left side sample count matches the right side sample count for the previous knot.
            // The middle sample count is the number of samples between knots (same as ComputeCurveSamples).
            float sampleRate[3] = float[3](0.0, 0.0, 0.0);
            if (samplesData.uInterval > 0.0)
            {{
                float minRate = 1.0 / samplesData.uInterval;
                if (samplesData.uOrder < 3)
                {{
                    // It's a plane or point, so just do the minimum sample.
                    sampleRate = float[3](minRate, minRate, minRate);
                }}
                else
                {{
                    float sampleRateLeft[{maxOrder}];
                    float sampleRateRight[{maxOrder}];

                    for (int k = 0; k < samplesData.uOrder; k++)
                    {{
                        sampleRateLeft[k] = 0.0;
                        sampleRateRight[k] = 0.0;
                    }}
                    for (int j = samplesData.vKnot-samplesData.vOrder; j < samplesData.vKnot; j++)
                    {{
                        int i = max(samplesData.uKnot - 1 - samplesData.uOrder, 0);
                        int iLimit = min(samplesData.uKnot - 1, samplesData.uN - 2);
                        int coefficientOffset = samplesData.uN*j + i;
                        vec3 coefficient0 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                        coefficientOffset++;
                        vec3 coefficient1 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                        float gap = texelFetch(uKnots, header + i+samplesData.uOrder).x - texelFetch(uKnots, header + i+1).x; // uKnots[i+uOrder] - uKnots[i+1]
                        vec3 dPoint0 = ((samplesData.uOrder - 1) / gap) * (coefficient1 - coefficient0);
                        while (i < iLimit)
                        {{
                            coefficientOffset++;
                            vec3 coefficient2 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                            gap = texelFetch(uKnots, header + i+1+samplesData.uOrder).x - texelFetch(uKnots, header + i+2).x; // uKnots[i+1+uOrder] - uKnots[i+2]
                            vec3 dPoint1 = ((samplesData.uOrder - 1) / gap) * (coefficient2 - coefficient1);
                            gap = texelFetch(uKnots, header + i+samplesData.uOrder).x - texelFetch(uKnots, header + i+2).x; // uKnots[i+uOrder] - uKnots[i+2]
                            vec3 d2Point = ((samplesData.uOrder - 2) / gap) * (dPoint1 - dPoint0);

                            int k = i - samplesData.uKnot + 1 + samplesData.uOrder;
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
                    for (int k = 1; k < samplesData.uOrder - 1; k++)
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
            uSamples[0] = min(floor(0.5 + samplesData.uInterval * sampleRate[0]), maxSamples);
            uSamples[1] = min(floor(0.5 + samplesData.uInterval * sampleRate[1]), maxSamples);
            uSamples[2] = min(floor(0.5 + samplesData.uInterval * sampleRate[2]), maxSamples);

            sampleRate = float[3](0.0, 0.0, 0.0);
            if (samplesData.vInterval > 0.0)
            {{
                float minRate = 1.0 / samplesData.vInterval;
                if (samplesData.vOrder < 3)
                {{
                    // It's a plane or point, so just do the minimum sample.
                    sampleRate = float[3](minRate, minRate, minRate);
                }}
                else
                {{
                    float sampleRateLeft[{maxOrder}];
                    float sampleRateRight[{maxOrder}];

                    for (int k = 0; k < samplesData.vOrder; k++)
                    {{
                        sampleRateLeft[k] = 0.0;
                        sampleRateRight[k] = 0.0;
                    }}
                    for (int i = samplesData.uKnot-samplesData.uOrder; i < samplesData.uKnot; i++)
                    {{
                        int j = max(samplesData.vKnot - 1 - samplesData.vOrder, 0);
                        int jLimit = min(samplesData.vKnot - 1, samplesData.vN - 2);
                        int coefficientOffset = samplesData.uN*j + i;
                        vec3 coefficient0 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                        coefficientOffset += samplesData.uN;
                        vec3 coefficient1 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                        float gap = texelFetch(uKnots, header + samplesData.uOrder+samplesData.uN + j+samplesData.vOrder).x - texelFetch(uKnots, header + samplesData.uOrder+samplesData.uN + j+1).x; // vKnots[j+vOrder] - vKnots[j+1]
                        vec3 dPoint0 = ((samplesData.vOrder - 1) / gap) * (coefficient1 - coefficient0);
                        while (j < jLimit)
                        {{
                            coefficientOffset += samplesData.uN;
                            vec3 coefficient2 = texelFetch(uXYZCoefs, coefficientOffset).xyz;
                            gap = texelFetch(uKnots, header + samplesData.uOrder+samplesData.uN + j+1+samplesData.vOrder).x - texelFetch(uKnots, header + samplesData.uOrder+samplesData.uN + j+2).x; // vKnots[j+1+vOrder] - vKnots[j+2]
                            vec3 dPoint1 = ((samplesData.vOrder - 1) / gap) * (coefficient2 - coefficient1);
                            gap = texelFetch(uKnots, header + samplesData.uOrder+samplesData.uN + j+samplesData.vOrder).x - texelFetch(uKnots, header + samplesData.uOrder+samplesData.uN + j+2).x; // vKnots[j+vOrder] - vKnots[j+2]
                            vec3 d2Point = ((samplesData.vOrder - 2) / gap) * (dPoint1 - dPoint0);

                            int k = j - samplesData.vKnot + 1 + samplesData.vOrder;
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
                    for (int k = 1; k < samplesData.vOrder - 1; k++)
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
            vSamples[0] = min(floor(0.5 + samplesData.vInterval * sampleRate[0]), maxSamples);
            vSamples[1] = min(floor(0.5 + samplesData.vInterval * sampleRate[1]), maxSamples);
            vSamples[2] = min(floor(0.5 + samplesData.vInterval * sampleRate[2]), maxSamples);
        }}
    """

    surfaceTCShaderCode = """
        #version 410 core

        layout (vertices = 1) out;

        const int header = 4;

        struct SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }};
        in SplineInfo vertexData[];

        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform samplerBuffer uKnots;
        uniform samplerBuffer uXYZCoefs;

        patch out SplineInfo tcData;

        {computeSampleRateCode}

        {computeSurfaceSamplesCode}

        void main()
        {{
            tcData = vertexData[gl_InvocationID];

            float uSamples[3];
            float vSamples[3];
            ComputeSurfaceSamples(gl_MaxTessGenLevel, tcData, uSamples, vSamples);
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

        struct SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }};
        patch in SplineInfo tcData;

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec4 uFillColor;
        uniform samplerBuffer uKnots;
        uniform samplerBuffer uXYZCoefs;
        uniform samplerBuffer uColorCoefs;

        flat out SplineInfo teData;
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
            parameters.x = tcData.u + gl_TessCoord.x * tcData.uInterval;
            ComputeBSpline(header, tcData.uOrder, tcData.uN, tcData.uKnot, parameters.x, uBSpline, duBSpline);

            float vBSpline[{maxOrder}];
            float dvBSpline[{maxOrder}];
            parameters.y = tcData.v + gl_TessCoord.y * tcData.vInterval;
            ComputeBSpline(header + tcData.uOrder+tcData.uN, tcData.vOrder, tcData.vN, tcData.vKnot, parameters.y, vBSpline, dvBSpline);

            {splineColorDeclarations}

            vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
            vec3 duPoint = vec3(0.0, 0.0, 0.0);
            vec3 dvPoint = vec3(0.0, 0.0, 0.0);
            {initializeSplineColor}
            int j = (tcData.vKnot - tcData.vOrder) * tcData.uN;
            for (int vB = 0; vB < tcData.vOrder; vB++)
            {{
                int i = j + tcData.uKnot - tcData.uOrder;
                for (int uB = 0; uB < tcData.uOrder; uB++)
                {{
                    vec3 coefs = texelFetch(uXYZCoefs, i).xyz;
                    point.xyz += uBSpline[uB] * vBSpline[vB] * coefs;
                    duPoint += duBSpline[uB] * vBSpline[vB] * coefs;
                    dvPoint += uBSpline[uB] * dvBSpline[vB] * coefs;
                    {computeSplineColor}
                    i++;
                }}
                j += tcData.uN;
            }}
            {postProcessSplineColor}

            teData = tcData;

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

        struct SplineInfo
        {{
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        }};
        in SplineInfo vertexData[];

        uniform mat4 uProjectionMatrix;
        uniform vec3 uScreenScale;
        uniform vec4 uClipBounds;
        uniform vec4 uFillColor;
        uniform vec3 uLightDirection;
        uniform samplerBuffer uKnots;
        uniform samplerBuffer uXYZCoefs;
        uniform samplerBuffer uColorCoefs;

        out vec3 splineColor; // We restrict our output to color to reduce the number of components per vertex.

        SplineInfo geometryData; // We don't output geometryData (too many components per vertex), but we do use it in ComputeSurfaceSamples.

        {computeSampleRateCode}

        {computeSurfaceSamplesCode}

        {computeBSplineCode}

        void main() 
        {{
            float uFullSamples[3];
            float vFullSamples[3];

            geometryData = vertexData[0];
            int maxVertices = gl_MaxGeometryTotalOutputComponents / 7; // The number of output components per vertex is 7 = position.xyzw + splineColor.rgb
            ComputeSurfaceSamples(maxVertices, geometryData, uFullSamples, vFullSamples);

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

                {splineColorDeclarations}

                float uBSpline[{maxOrder}];
                float duBSpline[{maxOrder}];
                float uBSplineNext[{maxOrder}];
                float duBSplineNext[{maxOrder}];
                float vBSpline[{maxOrder}];
                float dvBSpline[{maxOrder}];
                float deltaU = geometryData.uInterval / uSamples;
                float deltaV = geometryData.vInterval / vSamples;
                float u = geometryData.u;
                ComputeBSpline(header, geometryData.uOrder, geometryData.uN, geometryData.uKnot, u, uBSpline, duBSpline);

                int jOffset = (geometryData.vKnot - geometryData.vOrder) * geometryData.uN;
                int iOffset = geometryData.uKnot - geometryData.uOrder;

                for (int uSample = 0; uSample < uSamples; uSample++)
                {{
                    float uNext = u + deltaU;
                    ComputeBSpline(header, geometryData.uOrder, geometryData.uN, geometryData.uKnot, uNext, uBSplineNext, duBSplineNext);
                        
                    float v = geometryData.v;
                    for (int vSample = 0; vSample <= vSamples; vSample++)
                    {{
                        ComputeBSpline(header + geometryData.uOrder+geometryData.uN, geometryData.vOrder, geometryData.vN, geometryData.vKnot, v, vBSpline, dvBSpline);

                        vec4 point = vec4(0.0, 0.0, 0.0, 1.0);
                        vec3 duPoint = vec3(0.0, 0.0, 0.0);
                        vec3 dvPoint = vec3(0.0, 0.0, 0.0);
                        {initializeSplineColor}
                        int j = jOffset;
                        for (int vB = 0; vB < geometryData.vOrder; vB++)
                        {{
                            int i = j + iOffset;
                            for (int uB = 0; uB < geometryData.uOrder; uB++)
                            {{
                                vec3 coefs = texelFetch(uXYZCoefs, i).xyz;
                                point.xyz += uBSpline[uB] * vBSpline[vB] * coefs;
                                duPoint += duBSpline[uB] * vBSpline[vB] * coefs;
                                dvPoint += uBSpline[uB] * dvBSpline[vB] * coefs;
                                {computeSplineColor}
                                i++;
                            }}
                            j += geometryData.uN;
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
                        for (int vB = 0; vB < geometryData.vOrder; vB++)
                        {{
                            int i = j + iOffset;
                            for (int uB = 0; uB < geometryData.uOrder; uB++)
                            {{
                                vec3 coefs = texelFetch(uXYZCoefs, i).xyz;
                                point.xyz += uBSplineNext[uB] * vBSpline[vB] * coefs;
                                duPoint += duBSplineNext[uB] * vBSpline[vB] * coefs;
                                dvPoint += uBSplineNext[uB] * dvBSpline[vB] * coefs;
                                {computeSplineColor}
                                i++;
                            }}
                            j += geometryData.uN;
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
     
        struct SplineInfo
        {
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        };
        flat in SplineInfo teData;
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
            bool line = (uOptions & (1 << 2)) > 0 && (pixelPer.x * (parameters.x - teData.uFirst) < 1.5 || pixelPer.x * (teData.uFirst + teData.uSpan - parameters.x) < 1.5);
            line = line || ((uOptions & (1 << 2)) > 0 && (pixelPer.y * (parameters.y - teData.vFirst) < 1.5 || pixelPer.y * (teData.vFirst + teData.vSpan - parameters.y) < 1.5));
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.x * (parameters.x - teData.u) < 1.5);
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.y * (parameters.y - teData.v) < 1.5);
            color = line ? uLineColor : ((uOptions & (1 << 1)) > 0 ? vec4(splineColor, uFillColor.a) : vec4(0.0, 0.0, 0.0, 0.0));
            color.rgb = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * color.rgb;
            if (color.a == 0.0)
                discard;
        }
    """

    trimmedSurfaceFragmentShaderCode = """
        #version 410 core
     
        struct SplineInfo
        {
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        };
        flat in SplineInfo teData;
        in vec3 worldPosition;
        in vec3 splineColor;
        in vec3 normal;
        in vec2 parameters;
        in vec2 pixelPer;

        uniform vec4 uFillColor;
        uniform vec4 uLineColor;
        uniform vec3 uLightDirection;
        uniform int uOptions;
        uniform sampler2D uTrimTextureMap;

        out vec4 color;
     
        void main()
        {
        	vec2 tex = vec2((parameters.x - teData.uFirst) / teData.uSpan, (parameters.y - teData.vFirst) / teData.vSpan);
            float specular = pow(abs(dot(normal, normalize(uLightDirection + worldPosition / length(worldPosition)))), 25.0);
            bool line = (uOptions & (1 << 2)) > 0 && (pixelPer.x * (parameters.x - teData.uFirst) < 1.5 || pixelPer.x * (teData.uFirst + teData.uSpan - parameters.x) < 1.5);
            line = line || ((uOptions & (1 << 2)) > 0 && (pixelPer.y * (parameters.y - teData.vFirst) < 1.5 || pixelPer.y * (teData.vFirst + teData.vSpan - parameters.y) < 1.5));
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.x * (parameters.x - teData.u) < 1.5);
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.y * (parameters.y - teData.v) < 1.5);
            color = line ? uLineColor : ((uOptions & (1 << 1)) > 0 ? vec4(splineColor, uFillColor.a) : vec4(0.0, 0.0, 0.0, 0.0));
            color.rgb = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * color.rgb;
            if (color.a * texture(uTrimTextureMap, tex).r == 0.0)
                discard;
        }
    """
 
    def __init__(self, *args, eye=(0.0, 0.0, 3.0), center=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), draw_func=None, **kw):
        OpenGLFrame.__init__(self, *args, **kw)

        self.draw_func = draw_func
        self.animating = False
        self.animate = 0 # Set to number of milliseconds before showing next frame (0 means no animation)
        self.frameCount = 0
        self.tessellationEnabled = True
        self.glInitialized = False
        
        self.origin = None
        self.button = None
        self.mode = self.ROTATE
        
        self.computeBSplineCode = self.computeBSplineCode.format(maxOrder=self.maxOrder)

        self.SetBackgroundColor(0.0, 0.2, 0.2)

        self.SetDefaultView(eye, center, up)
        self.ResetView()

        self.bind("<ButtonPress>", self.MouseDown)
        self.bind("<Motion>", self.MouseMove)
        self.bind("<ButtonRelease>", self.MouseUp)
        self.bind("<MouseWheel>", self.MouseWheel)
        self.bind("<Unmap>", self.Unmap)

    @staticmethod
    def compute_color_vector(r, g=None, b=None, a=None):
        """
        Return an float32 array with the specified color.

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

        Returns
        -------
        color : `numpy.array`
            The specified color as an array of 4 float32 values between 0 and 1.
        """
        if isinstance(r, (int, np.integer)):
            red = float(r) / 255.0
            green = red
            blue = red
            alpha = 1.0
        elif np.isscalar(r):
            red = r
            green = red
            blue = red
            alpha = 1.0
        elif isinstance(r[0], (int, np.integer)):
            red = float(r[0]) / 255.0
            green = float(r[1]) / 255.0
            blue = float(r[2]) / 255.0
            alpha = float(r[3]) / 255.0 if len(r) >= 4 else 1.0
        else:
            red = r[0]
            green = r[1]
            blue = r[2]
            alpha = r[3] if len(r) >= 4 else 1.0

        if isinstance(g, (int, np.integer)):
            green = float(g) / 255.0
        elif np.isscalar(g):
            green = g

        if isinstance(b, (int, np.integer)):
            blue = float(b) / 255.0
        elif np.isscalar(b):
            blue = b

        if isinstance(a, (int, np.integer)):
            alpha = float(a) / 255.0
        elif np.isscalar(a):
            alpha = a
        
        return np.array((red, green, blue, alpha), np.float32)

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
        self.backgroundColor = self.compute_color_vector(r, g, b, a)
        if self.glInitialized:
            glClearColor(self.backgroundColor[0], self.backgroundColor[1], self.backgroundColor[2], self.backgroundColor[3])
    
    def ResetView(self):
        """
        Update the view position to default values.
        """
        self.eye = self.defaultEye.copy()
        self.look = self.defaultEye - self.defaultCenter
        self.anchorDistance = np.linalg.norm(self.look)
        self.anchorDistance = max(self.anchorDistance, 0.01)
        self.speed = 0.033 * self.anchorDistance
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
        if self.glInitialized:
            return
        
        #print("GL_VERSION: ", glGetString(GL_VERSION))
        #print("GL_SHADING_LANGUAGE_VERSION: ", glGetString(GL_SHADING_LANGUAGE_VERSION))
        #print("GL_MAX_TESS_GEN_LEVEL: ", glGetIntegerv(GL_MAX_TESS_GEN_LEVEL))
        glMajorVersion = glGetIntegerv(GL_MAJOR_VERSION)
        glMinorVersion = glGetIntegerv(GL_MINOR_VERSION)
        if glMajorVersion < 3 or (glMajorVersion == 3 and glMinorVersion < 3):
            print(f"OpenGL version must be 3.3 or higher. Current version is {glGetString(GL_VERSION)}")
            exit()
        elif glMajorVersion < 4 or (glMajorVersion == 4 and glMinorVersion < 1):
            self.tessellationEnabled = False # OpenGL version must be 4.1 or higher for tesselation
        else:
            self.tessellationEnabled = True

        if self.tessellationEnabled:
            # Set up frameBuffer into which we draw surface trims.    
            self.frameBuffer = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)

            # Create the texture buffer for surface trims.
            glActiveTexture(GL_TEXTURE4)
            self.trimTextureBuffer = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.trimTextureBuffer)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 512, 512, 0, GL_RED, GL_UNSIGNED_BYTE, None)

            # Attach trim texture buffer to framebuffer and validate framebuffer.
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.trimTextureBuffer, 0)
            glDrawBuffers(1, (GL_COLOR_ATTACHMENT0,))
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise ValueError("Framebuffer incomplete")
            
            # Set framebuffer back to default.
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Set up GL texture buffers for spline data
        # Knots data
        glActiveTexture(GL_TEXTURE0)
        self.knotsBuffer = glGenBuffers(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.knotsBuffer)
        glBindTexture(GL_TEXTURE_BUFFER, glGenTextures(1))
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.knotsBuffer)
        glBufferData(GL_TEXTURE_BUFFER, 4 * self.maxKnots, None, GL_STATIC_READ) # Each knot is a float (4 bytes)

        # XYZ coefficients data
        glActiveTexture(GL_TEXTURE1)
        self.xyzCoefsBuffer = glGenBuffers(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.xyzCoefsBuffer)
        glBindTexture(GL_TEXTURE_BUFFER, glGenTextures(1))
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, self.xyzCoefsBuffer)
        glBufferData(GL_TEXTURE_BUFFER, 4 * 3 * self.maxCoefficients, None, GL_STATIC_READ) # Each xyz coefficient is 3 floats (12 bytes)

        # Color coefficients data
        glActiveTexture(GL_TEXTURE2)
        self.colorCoefsBuffer = glGenBuffers(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.colorCoefsBuffer)
        glBindTexture(GL_TEXTURE_BUFFER, glGenTextures(1))
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, self.colorCoefsBuffer)
        glBufferData(GL_TEXTURE_BUFFER, 4 * 3 * self.maxCoefficients, None, GL_STATIC_READ) # Each color coefficient is 3 floats (12 bytes)

        # Set light direction
        self.lightDirection = np.array((0.63960218, 0.63960218, 0.42640144), np.float32)
        self.lightDirection = self.lightDirection / np.linalg.norm(self.lightDirection)

        # Compile shaders and link programs
        try:
            self.curveProgram = CurveProgram(self)
            self.surface3Program = SurfaceProgram(self, False, 3, "", "", "", "splineColor = uFillColor.rgb;")
            self.surface4Program = SurfaceProgram(self, False, 4,
                """
                    vec4 kVec = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 pVec;
                """, "splineColor = vec3(0.0, 0.0, 0.0);",
                "splineColor.r += uBSpline[uB] * vBSpline[vB] * texelFetch(uColorCoefs, i).x;",
                # Taken from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
                # uFillColor is passed in as HSV
                """
                    pVec = abs(fract(uFillColor.xxx + kVec.xyz) * 6.0 - kVec.www);
                    splineColor = uFillColor.z * mix(kVec.xxx, clamp(pVec - kVec.xxx, 0.0, 1.0), splineColor.r);
                """)
            self.surface6Program = SurfaceProgram(self, False, 6, "", "splineColor = vec3(0.0, 0.0, 0.0);",
                "splineColor += uBSpline[uB] * vBSpline[vB] * texelFetch(uColorCoefs, i).rgb;", "")

            if self.tessellationEnabled:
                self.trimmedSurface3Program = SurfaceProgram(self, True, 3, "", "", "", "splineColor = uFillColor.rgb;")
                self.trimmedSurface4Program = SurfaceProgram(self, True, 4,
                    """
                        vec4 kVec = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                        vec3 pVec;
                    """, "splineColor = vec3(0.0, 0.0, 0.0);",
                    "splineColor.r += uBSpline[uB] * vBSpline[vB] * texelFetch(uColorCoefs, i).x;",
                    # Taken from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
                    # uFillColor is passed in as HSV
                    """
                        pVec = abs(fract(uFillColor.xxx + kVec.xyz) * 6.0 - kVec.www);
                        splineColor = uFillColor.z * mix(kVec.xxx, clamp(pVec - kVec.xxx, 0.0, 1.0), splineColor.r);
                    """)
                self.trimmedSurface6Program = SurfaceProgram(self, True, 6, "", "splineColor = vec3(0.0, 0.0, 0.0);",
                    "splineColor += uBSpline[uB] * vBSpline[vB] * texelFetch(uColorCoefs, i).rgb;", "")

        except shaders.ShaderCompilationError as exception:
            error = exception.args[0]
            lineNumber = error.split(":")[3]
            source = exception.args[1][0]
            badLine = source.split(b"\n")[int(lineNumber)-1]
            shaderType = exception.args[2]
            print(shaderType, error)
            print(badLine)
            quit()

        # Set default draw parameters.
        glUseProgram(0)
        glEnable( GL_DEPTH_TEST )
        glClearColor(self.backgroundColor[0], self.backgroundColor[1], self.backgroundColor[2], self.backgroundColor[3])

    def ResetBounds(self):
        """
        Handle window size and/or clipping plane update (typically after a window resize).
        """
        if not self.glInitialized:
            return
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        xExtent = self.width / self.height
        defaultAnchorDistance = np.linalg.norm(self.defaultEye - self.defaultCenter)
        clipDistance = defaultAnchorDistance / np.sqrt(3.0)
        near = 0.01 * defaultAnchorDistance / 3.0
        far = 3.0 * defaultAnchorDistance
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
        if self.tessellationEnabled:
            self.trimmedSurface3Program.ResetBounds(self)
            self.trimmedSurface4Program.ResetBounds(self)
            self.trimmedSurface6Program.ResetBounds(self)

        glUseProgram(0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def redraw(self):
        """
        Handle `OpenGLFrame` redraw action. Updates view and draws spline list.
        """
        if not self.glInitialized:
            return
        
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

        if self.draw_func is not None:
            self.draw_func(self, transform)

        glFlush()

        if self.animate > 0:
            self.frameCount = (self.frameCount + 1) % 1000000

    def Unmap(self, event):
        """
        Handle window unmap.
        """
        self.glInitialized = False

    def Update(self):
        """
        Update the frame, typically after updating the spline list.
        """
        try:
            self.tkExpose(None)
        except AttributeError:
            pass

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
        Set anchor distance and/or flying speed (depending on mode).

        Parameters
        ----------
        scale : `float`
            Scale between 0 and 1.
        """
        if self.mode == self.FLY:
            self.speed = 0.033 * self.anchorDistance * (100.0 ** float(scale) - 1.0) / 99.0
        else:
            defaultAnchorDistance = np.linalg.norm(self.defaultEye - self.defaultCenter)
            self.anchorDistance = 2.0 * float(scale) * defaultAnchorDistance
            self.anchorDistance = max(self.anchorDistance, 0.01)
            self.speed = 0.033 * self.anchorDistance
            self.eye = self.anchorPosition + self.anchorDistance * self.look
            self.Update()
    
    def SetAnimating(self, animating):
        self.animating = animating
        if self.animating:
            self.animate = self.MsPerFrame
        elif self.mode != self.FLY or self.button is None:
            self.animate = 0 # Stop animating

    def MouseDown(self, event):
        """
        Handle mouse down event.
        """
        self.origin = np.array((event.x, event.y), np.float32)
        self.current = self.origin
        self.button = event.num

        if self.button == 4 or self.button == 5: # MouseWheel
            self.anchorDistance *= 0.9 if self.button == 4 else 1.1
            self.anchorDistance = max(self.anchorDistance, 0.01)
            self.speed = 0.033 * self.anchorDistance
            self.eye = self.anchorPosition + self.anchorDistance * self.look
            self.Update()
        
        if self.mode == self.FLY and not self.animating:
            self.animate = self.MsPerFrame
            self.Update()

    def MouseMove(self, event):
        """
        Handle mouse move event.
        """
        self.current = np.array((event.x, event.y), np.float32)
        if self.button is not None and (self.mode == self.ROTATE or self.mode == self.PAN):
            self.Update()

    def MouseUp(self, event):
        """
        Handle mouse up event.
        """
        self.origin = None
        self.button = None
        if self.mode == self.FLY and not self.animating:
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
        self.anchorDistance = max(self.anchorDistance, 0.01)
        self.speed = 0.033 * self.anchorDistance
        self.eye = self.anchorPosition + self.anchorDistance * self.look
        self.Update()

    @staticmethod
    def make_drawable(spline):
        """
        Ensure a `Spline` can be drawn in a `SplineOpenGLFrame`. Converts 1D splines into 3D drawable curves, 
        2D splines into drawable surfaces (y-axis hold amplitude), and 3D splines into drawable solids.

        Spline surfaces and solids with more than 3 dependent variables will have their added dimensions rendered 
        as colors (up to 6 dependent variables are supported).
        """
        if not(isinstance(spline, Spline)): raise ValueError("Invalid spline")
        if spline.nInd > 3: raise ValueError("Spline must have no more than 3 independent variables")
        if spline.nDep > 6: raise ValueError("Spline must have no more than 6 dependent variables")

        if not hasattr(spline, "cache"):
            spline.cache = {}
        
        if not "knots32" in spline.cache:
            knotList = [knots.astype(np.float32, copy=False) for knots in spline.knots]
            spline.cache["knots32"] = knotList  # Shaders expect float32 knots

        if not "xyzCoefs32" in spline.cache:
            xyzCoefs = np.zeros((3, *spline.nCoef), np.float32)
            # Curves
            if spline.nInd == 1:
                if spline.nDep == 1:
                    graph = spline.graph()
                    xyzCoefs[0] = graph.coefs[0]
                    xyzCoefs[1] = graph.coefs[1]
                else:
                    xyzCoefs[:min(spline.nDep, 3)] = spline.coefs[:min(spline.nDep, 3)]
            # Surfaces and Solids
            elif 2 <= spline.nInd <= 3:
                if spline.nDep == 1:
                    graph = spline.graph()
                    xyzCoefs[0] = graph.coefs[0]
                    xyzCoefs[1] = graph.coefs[2]
                    xyzCoefs[2] = graph.coefs[1]
                else:
                    xyzCoefs[:min(spline.nDep, 3)] = spline.coefs[:min(spline.nDep, 3)]
                    # For dimensions above three, rescale dependent variables to [0, 1].
                    if spline.nDep > 3:
                        colorCoefs = np.zeros((3, *spline.nCoef), np.float32)
                        for i in range(3, spline.nDep):
                            minCoef = spline.coefs[i].min()
                            rangeCoef = spline.coefs[i].max() - minCoef
                            if rangeCoef > 1.0e-8:
                                colorCoefs[i-3] = (spline.coefs[i] - minCoef) / rangeCoef
                            else:
                                colorCoefs[i-3] = max(0.0, min(1.0, minCoef))
                        spline.cache["colorCoefs32"] = colorCoefs.T # Shaders expect transpose of float32 coefs
            else:
                raise ValueError("Can't convert to drawable spline.")
        
            spline.cache["xyzCoefs32"] = xyzCoefs.T # Shaders expect transpose of float32 coefs
    
        if not "fillColor" in spline.metadata:
            spline.metadata["fillColor"] = np.array((0.0, 1.0, 0.0, 1.0), np.float32)
        if not "lineColor" in spline.metadata:
            spline.metadata["lineColor"] = np.array((0.0, 0.0, 0.0, 1.0) if spline.nInd > 1 else (1.0, 1.0, 1.0, 1.0), np.float32)
        if not "options" in spline.metadata:
            spline.metadata["options"] = SplineOpenGLFrame.SHADED | SplineOpenGLFrame.BOUNDARY
        if not "animate" in spline.metadata:
            spline.metadata["animate"] = None

    def tessellate2DSolid(self, solid):
        """
        Returns an array of triangles that tessellate the given 2D solid
        """
        assert solid.dimension == 2
        assert solid.containsInfinity == False

        if not self.tessellationEnabled:
            return None

        # First, collect all manifold contour endpoints, accounting for slight numerical error.
        class Endpoint:
            def __init__(self, curve, t, clockwise, isStart, otherEnd=None):
                self.curve = curve
                self.t = t
                self.xy = curve.manifold.evaluate((t,))
                self.clockwise = clockwise
                self.isStart = isStart
                self.otherEnd = otherEnd
                self.connection = None
        endpoints = []
        for curve in solid.boundaries:
            curve.trim.boundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), -boundary.manifold.normal(0.0)))
            leftB = 0
            rightB = 0
            boundaryCount = len(curve.trim.boundaries)
            while leftB < boundaryCount:
                if curve.trim.boundaries[leftB].manifold.normal(0.0) < 0.0:
                    leftPoint = curve.trim.boundaries[leftB].manifold.evaluate(0.0)[0]
                    while rightB < boundaryCount:
                        rightPoint = curve.trim.boundaries[rightB].manifold.evaluate(0.0)[0]
                        if leftPoint - Manifold.minSeparation < rightPoint and curve.trim.boundaries[rightB].manifold.normal(0.0) > 0.0:
                            t = curve.manifold.tangent_space(leftPoint)[:,0]
                            n = curve.manifold.normal(leftPoint)
                            clockwise = t[0] * n[1] - t[1] * n[0] > 0.0
                            ep1 = Endpoint(curve, leftPoint, clockwise, rightPoint >= leftPoint)
                            ep2 = Endpoint(curve, rightPoint, clockwise, rightPoint < leftPoint, ep1)
                            ep1.otherEnd = ep2
                            endpoints.append(ep1)
                            endpoints.append(ep2)
                            leftB = rightB
                            rightB += 1
                            break
                        rightB += 1
                leftB += 1

        # Second, collect all valid pairings of endpoints (normal not negated between segments).
        Connection = namedtuple('Connection', ('distance', 'ep1', 'ep2'))
        connections = []
        for i, ep1 in enumerate(endpoints[:-1]):
            for ep2 in endpoints[i+1:]:
                if (ep1.clockwise == ep2.clockwise and ep1.isStart != ep2.isStart) or \
                    (ep1.clockwise != ep2.clockwise and ep1.isStart == ep2.isStart):
                    connections.append(Connection(np.linalg.norm(ep1.xy - ep2.xy), ep1, ep2))

        # Third, only keep closest pairings (prune the rest).
        connections.sort(key=lambda connection: -connection.distance)
        while connections:
            connection = connections.pop()
            connection.ep1.connection = connection.ep2
            connection.ep2.connection = connection.ep1
            connections = [c for c in connections if c.ep1 is not connection.ep1 and c.ep1 is not connection.ep2 and \
                    c.ep2 is not connection.ep1 and c.ep2 is not connection.ep2]
            
        # Fourth, set up GLUT to tesselate the solid.
        tess = gluNewTess()
        gluTessProperty(tess, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_ODD)
        vertices = []
        def beginCallback(type=None):
            vertices = []
        def edgeFlagDataCallback(flag, polygonData):
            pass # Forces triangulation of polygons rather than triangle fans or strips
        def vertexCallback(vertex, otherData=None):
            vertices.append(vertex[:2])
        def combineCallback(vertex, neighbors, neighborWeights, outData=None):
            outData = vertex
            return outData
        def endCallback():
            pass
        gluTessCallback(tess, GLU_TESS_BEGIN, beginCallback)
        gluTessCallback(tess, GLU_TESS_EDGE_FLAG_DATA, edgeFlagDataCallback)
        gluTessCallback(tess, GLU_TESS_VERTEX, vertexCallback)
        gluTessCallback(tess, GLU_TESS_COMBINE, combineCallback)
        gluTessCallback(tess, GLU_TESS_END, endCallback)

        # Fifth, trace the contours from pairing to pairing, using GLUT to tesselate the interior.
        gluTessBeginPolygon(tess, 0)
        while endpoints:
            start = endpoints[0]
            if not start.isStart:
                start = start.otherEnd
            # Run backwards until you hit start again or hit an end.
            if start.connection is not None:
                originalStart = start
                next = start.connection
                start = None
                while next is not None and start is not originalStart:
                    start = next.otherEnd
                    next = start.connection
            # Run forwards submitting vertices for the contour.
            next = start
            gluTessBeginContour(tess)
            while next is not None:
                endpoints.remove(next)
                endpoints.remove(next.otherEnd)
                subdivisions = max(int(abs(next.otherEnd.t - next.t) / 0.1), 20) if isinstance(next.curve.manifold, Spline) else 2
                for t in np.linspace(next.t, next.otherEnd.t, subdivisions):
                    xy = next.curve.manifold.evaluate((t,))
                    vertex = (*xy, 0.0)
                    gluTessVertex(tess, vertex, vertex)
                next = next.otherEnd.connection
                if next is start:
                    break
            gluTessEndContour(tess)
        gluTessEndPolygon(tess)
        gluDeleteTess(tess)
        return np.array(vertices, np.float32)

    def _DrawPoints(self, spline, drawCoefficients):
        """
        Draw spline points for an nInd == 0 or order == 1 spline within a `SplineOpenGLFrame`. The self will call this method for you.
        """
        glColor4fv(spline.metadata["lineColor"])
        glBegin(GL_POINTS)
        if spline.nInd == 0:
            glVertex3fv(drawCoefficients)
        else:
            for point in drawCoefficients:
                glVertex3fv(point)
        glEnd()

    def _DrawCurve(self, spline, drawCoefficients):
        """
        Draw a spline curve (nInd == 1) within a `SplineOpenGLFrame`. The self will call this method for you.
        """
        if spline.metadata["options"] & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            glBegin(GL_LINE_STRIP)
            for point in drawCoefficients:
                glVertex3f(point[0], point[1], point[2])
            glEnd()

        # Load spline data into textures
        glActiveTexture(GL_TEXTURE0)
        glBindBuffer(GL_TEXTURE_BUFFER, self.knotsBuffer)
        offset = 0
        size = 4 * 2
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((spline.order[0], spline.nCoef[0]), np.float32))
        offset += size
        knots = spline.cache["knots32"]
        size = 4 * len(knots[0])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, knots[0])
        glActiveTexture(GL_TEXTURE1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.xyzCoefsBuffer)
        size = 4 * 3 * spline.nCoef[0]
        glBufferSubData(GL_TEXTURE_BUFFER, 0, size, drawCoefficients)

        # Render spline
        program = self.curveProgram
        glUseProgram(program.curveProgram)
        glUniform4fv(program.uCurveLineColor, 1, spline.metadata["lineColor"])
        if self.tessellationEnabled:
            glPatchParameteri(GL_PATCH_VERTICES, 1)
            glDrawArraysInstanced(GL_PATCHES, 0, 1, spline.nCoef[0] - spline.order[0] + 1)
        else:
            glDrawArraysInstanced(GL_POINTS, 0, 1, spline.nCoef[0] - spline.order[0] + 1)
            glFlush() # Old graphics card
        glUseProgram(0)

    @staticmethod
    def ConvertRGBToHSV(r, g, b, a):
        # Taken from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
        K = 0.0
        if g < b:
            tmp = g
            g = b
            b = tmp
            K = -1.0
        if r < g:
            tmp = r
            r = g
            g = tmp
            K = -2.0 / 6.0 - K
        chroma = r - min(g, b)
        return np.array((abs(K + (g - b) / (6.0 * chroma + 1e-20)), chroma / (r + 1e-20), r, a), np.float32)
    
    def _DrawSurface(self, spline, drawCoefficients):
        """
        Draw a spline surface (nInd == 2) within a `SplineOpenGLFrame`.
        """
        if spline.metadata["options"] & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            for pointList in drawCoefficients:
                glBegin(GL_LINE_STRIP)
                for point in pointList:
                    glVertex3f(point[0], point[1], point[2])
                glEnd()

        fillColor = spline.metadata["fillColor"]
        if spline.nDep <= 3:
            nDep = 3
            program = self.trimmedSurface3Program if "trim" in spline.cache else self.surface3Program
        elif spline.nDep == 4:
            nDep = 4
            program = self.trimmedSurface4Program if "trim" in spline.cache else self.surface4Program
            program = self.surface4Program
            fillColor = self.ConvertRGBToHSV(fillColor[0], fillColor[1], fillColor[2], fillColor[3])
        elif spline.nDep <= 6:
            nDep = 6
            program = self.trimmedSurface6Program if "trim" in spline.cache else self.surface6Program
        else:
            raise ValueError("Can't draw surface.")
        
        useBlending = fillColor[3] < 1.0
        if useBlending:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable( GL_BLEND )
            glDisable( GL_DEPTH_TEST )

        # Load spline data into textures
        glActiveTexture(GL_TEXTURE0)
        glBindBuffer(GL_TEXTURE_BUFFER, self.knotsBuffer)
        offset = 0
        size = 4 * 4
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((spline.order[0], spline.order[1], spline.nCoef[0], spline.nCoef[1]), np.float32))
        offset += size
        knots = spline.cache["knots32"]
        size = 4 * len(knots[0])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, knots[0])
        offset += size
        size = 4 * len(knots[1])
        glBufferSubData(GL_TEXTURE_BUFFER, offset, size, knots[1])
        glActiveTexture(GL_TEXTURE1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.xyzCoefsBuffer)
        size = 4 * 3 * spline.nCoef[0] * spline.nCoef[1]
        glBufferSubData(GL_TEXTURE_BUFFER, 0, size, drawCoefficients)
        if nDep > 3:
            glActiveTexture(GL_TEXTURE2)
            glBindBuffer(GL_TEXTURE_BUFFER, self.colorCoefsBuffer)
            glBufferSubData(GL_TEXTURE_BUFFER, 0, size, spline.cache["colorCoefs32"])

        # Render spline
        glUseProgram(program.surfaceProgram)
        glUniform4fv(program.uSurfaceFillColor, 1, fillColor)
        glUniform4fv(program.uSurfaceLineColor, 1, spline.metadata["lineColor"])
        glUniform1i(program.uSurfaceOptions, spline.metadata["options"])
        if self.tessellationEnabled:
            glPatchParameteri(GL_PATCH_VERTICES, 1)
            glDrawArraysInstanced(GL_PATCHES, 0, 1, (spline.nCoef[0] - spline.order[0] + 1) * (spline.nCoef[1] - spline.order[1] + 1))
        else:
            glDrawArraysInstanced(GL_POINTS, 0, 1, (spline.nCoef[0] - spline.order[0] + 1) * (spline.nCoef[1] - spline.order[1] + 1))
            glFlush() # Old graphics card
        glUseProgram(0)
        if useBlending:
            glDisable( GL_BLEND )
            glEnable( GL_DEPTH_TEST )
    
    def _DrawSolid(self, spline, drawCoefficients):
        """
        Draw a spline solid (nInd == 3) within a `SplineOpenGLFrame`.
        """
        if spline.metadata["options"] & self.HULL:
            glColor3f(0.0, 0.0, 1.0)
            for pointSet in drawCoefficients:
                for pointList in pointSet:
                    glBegin(GL_LINE_STRIP)
                    for point in pointList:
                        glVertex3f(point[0], point[1], point[2])
                    glEnd()

        fillColor = spline.metadata["fillColor"].copy()
        lineColor = spline.metadata["lineColor"].copy()
        if spline.nDep <= 3:
            nDep = 3
            program = self.trimmedSurface3Program if "trim" in spline.cache else self.surface3Program
        elif spline.nDep == 4:
            nDep = 4
            program = self.trimmedSurface4Program if "trim" in spline.cache else self.surface4Program
            program = self.surface4Program
            fillColor = self.ConvertRGBToHSV(fillColor[0], fillColor[1], fillColor[2], fillColor[3])
        elif spline.nDep <= 6:
            nDep = 6
            program = self.trimmedSurface6Program if "trim" in spline.cache else self.surface6Program
        else:
            raise ValueError("Can't draw surface.")
        fillColor[3] *= 0.5
        lineColor[3] *= 0.5
        knots = spline.cache["knots32"]

        def _DrawBoundarySurface(axis, index):
            fullSlice = slice(None)
            if axis == 0:
                i1 = 1
                i2 = 2
                coefSlice = (fullSlice, fullSlice, index, fullSlice)
            elif axis == 1:
                i1 = 0
                i2 = 2
                coefSlice = (fullSlice, index, fullSlice, fullSlice)
            else:
                i1 = 0
                i2 = 1
                coefSlice = (index, fullSlice, fullSlice, fullSlice)

            # Load spline data into textures
            glActiveTexture(GL_TEXTURE0)
            glBindBuffer(GL_TEXTURE_BUFFER, self.knotsBuffer)
            offset = 0
            size = 4 * 4
            glBufferSubData(GL_TEXTURE_BUFFER, offset, size, np.array((spline.order[i1], spline.order[i2], spline.nCoef[i1], spline.nCoef[i2]), np.float32))
            offset += size
            size = 4 * len(knots[i1])
            glBufferSubData(GL_TEXTURE_BUFFER, offset, size, knots[i1])
            offset += size
            size = 4 * len(knots[i2])
            glBufferSubData(GL_TEXTURE_BUFFER, offset, size, knots[i2])
            glActiveTexture(GL_TEXTURE1)
            glBindBuffer(GL_TEXTURE_BUFFER, self.xyzCoefsBuffer)
            size = 4 * 3 * spline.nCoef[i1] * spline.nCoef[i2]
            glBufferSubData(GL_TEXTURE_BUFFER, 0, size, drawCoefficients[coefSlice])
            if nDep > 3:
                glActiveTexture(GL_TEXTURE2)
                glBindBuffer(GL_TEXTURE_BUFFER, self.colorCoefsBuffer)
                glBufferSubData(GL_TEXTURE_BUFFER, 0, size, spline.cache["colorCoefs32"][coefSlice])

            # Render spline
            glUseProgram(program.surfaceProgram)
            if self.tessellationEnabled:
                glPatchParameteri(GL_PATCH_VERTICES, 1)
                glDrawArraysInstanced(GL_PATCHES, 0, 1, (spline.nCoef[i1] - spline.order[i1] + 1) * (spline.nCoef[i2] - spline.order[i2] + 1))
            else:
                glDrawArraysInstanced(GL_POINTS, 0, 1, (spline.nCoef[i1] - spline.order[i1] + 1) * (spline.nCoef[i2] - spline.order[i2] + 1))
                glFlush() # Old graphics card

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glDisable( GL_DEPTH_TEST )
        glUseProgram(program.surfaceProgram)
        glUniform4fv(program.uSurfaceFillColor, 1, fillColor)
        glUniform4fv(program.uSurfaceLineColor, 1, lineColor)
        glUniform1i(program.uSurfaceOptions, spline.metadata["options"])

        _DrawBoundarySurface(0, 0)
        _DrawBoundarySurface(0, -1)
        _DrawBoundarySurface(1, 0)
        _DrawBoundarySurface(1, -1)
        _DrawBoundarySurface(2, 0)
        _DrawBoundarySurface(2, -1)

        glUseProgram(0)
        glDisable( GL_BLEND )
        glEnable( GL_DEPTH_TEST )

    def DrawSpline(self, spline, transform):
        """
        Draw a spline within a `SplineOpenGLFrame`.
        """
        # Fill trim stencil.
        if "trim" in spline.cache:
            # Draw trim tessellation into trim texture framebuffer.
            glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)
            glDisable(GL_DEPTH_TEST)
            glViewport(0,0,512,512)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            bounds = spline.domain()
            glOrtho(bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1], -1.0, 1.0)
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_TRIANGLES)
            for vertex in spline.cache["trim"]:
                glVertex2fv(vertex)
            glEnd()
            glFlush()
            # Reset view for main framebuffer.
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glEnable(GL_DEPTH_TEST)
            glViewport(0, 0, self.width, self.height)
            glClearColor(self.backgroundColor[0], self.backgroundColor[1], self.backgroundColor[2], self.backgroundColor[3])
            glMatrixMode(GL_PROJECTION)
            glLoadMatrixf(self.projection)
            glMatrixMode(GL_MODELVIEW)

        # Retrieve transposed float32 xyz coefficients.
        xyzCoefs = spline.cache["xyzCoefs32"]

        # Contract spline if it's animating.
        nInd = spline.metadata["animate"]
        if nInd is not None:
            # Contraction value is set to cycle every 10 seconds (10000 ms).
            u1 = spline.knots[nInd][spline.order[nInd] - 1]
            u2 = spline.knots[nInd][spline.nCoef[nInd]]
            u = u1 + 0.49999 * (u2 - u1) * (1.0 - np.cos(2.0 * np.pi * self.frameCount * self.MsPerFrame / 10000))
            # Contract spline.
            knots = spline.cache["knots32"]
            coefs = xyzCoefs if spline.nDep <= 3 else np.append(xyzCoefs, spline.cache["colorCoefs32"], axis=-1)
            ix, bValues = spline.bspline_values(None, knots[nInd], spline.order[nInd], u)
            coefs = np.moveaxis(coefs, spline.nInd - nInd - 1, -1) # Account for transpose
            coefs = coefs[..., ix - spline.order[nInd]:ix] @ bValues
            knots = [knots[i] for i in range(spline.nInd) if i != nInd]
            spline = type(spline)(spline.nInd - 1, coefs.shape[-1], 
                [spline.order[i] for i in range(spline.nInd) if i != nInd],
                [spline.nCoef[i] for i in range(spline.nInd) if i != nInd],
                knots, coefs.T, spline.metadata)
            xyzCoefs = coefs[..., :3]
            if spline.nDep <= 3:
                spline.cache = {"knots32": knots, "xyzCoefs32": xyzCoefs}
            else:
                spline.cache = {"knots32": knots, "xyzCoefs32": xyzCoefs, "colorCoefs32": coefs[..., 3:]}

        # Transform coefs by view transform.
        drawCoefficients = xyzCoefs @ transform[:3,:3] + transform[3,:3]

        # Draw spline.
        if spline.nInd == 0 or spline.order[0] == 1:
            self._DrawPoints(spline, drawCoefficients)
        elif spline.nInd == 1:
            self._DrawCurve(spline, drawCoefficients)
        elif spline.nInd == 2:
            self._DrawSurface(spline, drawCoefficients)
        elif spline.nInd == 3:
            self._DrawSolid(spline, drawCoefficients)

class CurveProgram:
    """ Compile curve program """
    def __init__(self, frame):
        if frame.tessellationEnabled:
            self.curveProgram = shaders.compileProgram(
                shaders.compileShader(frame.curveVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(frame.curveTCShaderCode.format(
                    computeSampleRateCode=frame.computeSampleRateCode,
                    computeCurveSamplesCode=frame.computeCurveSamplesCode), GL_TESS_CONTROL_SHADER),
                shaders.compileShader(frame.curveTEShaderCode.format(
                    computeBSplineCode=frame.computeBSplineCode,
                    maxOrder=frame.maxOrder), GL_TESS_EVALUATION_SHADER), 
                shaders.compileShader(frame.curveFragmentShaderCode, GL_FRAGMENT_SHADER),
                validate = False)
        else:
            self.curveProgram = shaders.compileProgram(
                shaders.compileShader(frame.curveVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(frame.curveGeometryShaderCode.format(
                    computeSampleRateCode=frame.computeSampleRateCode,
                    computeCurveSamplesCode=frame.computeCurveSamplesCode,
                    computeBSplineCode=frame.computeBSplineCode,
                    maxOrder=frame.maxOrder), GL_GEOMETRY_SHADER), 
                shaders.compileShader(frame.curveFragmentShaderCode, GL_FRAGMENT_SHADER))

        glUseProgram(self.curveProgram)
        self.uCurveProjectionMatrix = glGetUniformLocation(self.curveProgram, 'uProjectionMatrix')
        self.uCurveScreenScale = glGetUniformLocation(self.curveProgram, 'uScreenScale')
        self.uCurveClipBounds = glGetUniformLocation(self.curveProgram, 'uClipBounds')
        self.uCurveLineColor = glGetUniformLocation(self.curveProgram, 'uLineColor')
        glUniform1i(glGetUniformLocation(self.curveProgram, 'uKnots'), 0) # GL_TEXTURE0 is the knots buffer texture
        glUniform1i(glGetUniformLocation(self.curveProgram, 'uXYZCoefs'), 1) # GL_TEXTURE1 is the xyz coefs texture
    
    def ResetBounds(self, frame):
        """Reset bounds and other frame configuration for curve program"""
        glUseProgram(self.curveProgram)
        glUniformMatrix4fv(self.uCurveProjectionMatrix, 1, GL_FALSE, frame.projection)
        glUniform3fv(self.uCurveScreenScale, 1, frame.screenScale)
        glUniform4fv(self.uCurveClipBounds, 1, frame.clipBounds)

class SurfaceProgram:
    """ Compile surface program """
    def __init__(self, frame, trimmed, nDep, splineColorDeclarations, initializeSplineColor, computeSplineColor, postProcessSplineColor):
        if frame.tessellationEnabled:
            if trimmed:
                compiledFragmentShader = shaders.compileShader(frame.trimmedSurfaceFragmentShaderCode, GL_FRAGMENT_SHADER)
            else:
                compiledFragmentShader = shaders.compileShader(frame.surfaceFragmentShaderCode, GL_FRAGMENT_SHADER)
            self.surfaceProgram = shaders.compileProgram(
                shaders.compileShader(frame.surfaceVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(frame.surfaceTCShaderCode.format(
                    computeSampleRateCode=frame.computeSampleRateCode,
                    computeSurfaceSamplesCode=frame.computeSurfaceSamplesCode.format(maxOrder=frame.maxOrder)), GL_TESS_CONTROL_SHADER), 
                shaders.compileShader(frame.surfaceTEShaderCode.format(
                    computeBSplineCode=frame.computeBSplineCode,
                    splineColorDeclarations=splineColorDeclarations,
                    initializeSplineColor=initializeSplineColor,
                    computeSplineColor=computeSplineColor,
                    postProcessSplineColor=postProcessSplineColor,
                    maxOrder=frame.maxOrder), GL_TESS_EVALUATION_SHADER), 
                compiledFragmentShader,
                validate = False)
        else:
            self.surfaceProgram = shaders.compileProgram(
                shaders.compileShader(frame.surfaceVertexShaderCode, GL_VERTEX_SHADER), 
                shaders.compileShader(frame.surfaceGeometryShaderCode.format(
                    computeSampleRateCode=frame.computeSampleRateCode,
                    computeSurfaceSamplesCode=frame.computeSurfaceSamplesCode.format(maxOrder=frame.maxOrder),
                    computeBSplineCode=frame.computeBSplineCode,
                    splineColorDeclarations=splineColorDeclarations,
                    initializeSplineColor=initializeSplineColor,
                    computeSplineColor=computeSplineColor,
                    postProcessSplineColor=postProcessSplineColor,
                    maxOrder=frame.maxOrder), GL_GEOMETRY_SHADER), 
                shaders.compileShader(frame.surfaceSimpleFragmentShaderCode, GL_FRAGMENT_SHADER))

        # Initialize program parameters.
        glUseProgram(self.surfaceProgram)
        self.uSurfaceProjectionMatrix = glGetUniformLocation(self.surfaceProgram, 'uProjectionMatrix')
        self.uSurfaceScreenScale = glGetUniformLocation(self.surfaceProgram, 'uScreenScale')
        self.uSurfaceClipBounds = glGetUniformLocation(self.surfaceProgram, 'uClipBounds')
        self.uSurfaceFillColor = glGetUniformLocation(self.surfaceProgram, 'uFillColor')
        self.uSurfaceLineColor = glGetUniformLocation(self.surfaceProgram, 'uLineColor')
        glUniform3fv(glGetUniformLocation(self.surfaceProgram, 'uLightDirection'), 1, frame.lightDirection)
        self.uSurfaceOptions = glGetUniformLocation(self.surfaceProgram, 'uOptions')
        glUniform1i(glGetUniformLocation(self.surfaceProgram, 'uKnots'), 0) # GL_TEXTURE0 is the knots buffer texture
        glUniform1i(glGetUniformLocation(self.surfaceProgram, 'uXYZCoefs'), 1) # GL_TEXTURE1 is the xyz coefs texture
        if nDep > 3:
            glUniform1i(glGetUniformLocation(self.surfaceProgram, 'uColorCoefs'), 2) # GL_TEXTURE2 is the color coefs texture
        if trimmed and frame.tessellationEnabled:
            glUniform1i(glGetUniformLocation(self.surfaceProgram, 'uTrimTextureMap'), 4) # GL_TEXTURE4 is the trim texture map

    def ResetBounds(self, frame):
        """Reset bounds and other frame configuration for surface program"""
        glUseProgram(self.surfaceProgram)
        glUniformMatrix4fv(self.uSurfaceProjectionMatrix, 1, GL_FALSE, frame.projection)
        glUniform3fv(self.uSurfaceScreenScale, 1, frame.screenScale)
        glUniform4fv(self.uSurfaceClipBounds, 1, frame.clipBounds)