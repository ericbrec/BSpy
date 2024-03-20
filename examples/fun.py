import numpy as np
from os import path
from OpenGL.GL import *
from PIL import Image
from bspy import Spline, Viewer, SplineOpenGLFrame

class FunOpenGLFrame(SplineOpenGLFrame):

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
        in vec3 normal;
        in vec2 parameters;
        in vec2 pixelPer;

        uniform vec4 uFillColor;
        uniform vec4 uLineColor;
        uniform vec3 uLightDirection;
        uniform int uOptions;
        uniform sampler2D uTextureMap;

        out vec3 color;
     
        void main()
        {
            float specular = pow(abs(dot(normal, normalize(uLightDirection + worldPosition.xyz / length(worldPosition)))), 25.0);
            vec3 reflection = normalize(reflect(worldPosition.xyz, normal));
        	vec2 tex = vec2(0.5 * (1.0 - atan(reflection.x, reflection.z) / 3.1416), 0.5 * (1.0 - reflection.y));
            color = (uOptions & (1 << 2)) > 0 ? uFillColor.rgb : vec3(0.0, 0.0, 0.0);
            color = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * texture(uTextureMap, tex).rgb;

        	tex = vec2((parameters.x - inData.uFirst) / inData.uSpan, (parameters.y - inData.vFirst) / inData.vSpan);
            color = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * texture(uTextureMap, tex).rgb;
        }
    """

    def CreateGLResources(self):
        #img = Image.open(path.expanduser("~/OneDrive/Pictures/Backgrounds/2020 Cannon Beach.jpg"))
        img = Image.open(path.expanduser("~/OneDrive/Pictures/Backgrounds/Tom Sailing.jpg"))
        img_data = np.array((img.getdata())).astype(np.int8)
        self.textureBuffer = glGenTextures(1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.textureBuffer)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)            
        glActiveTexture(GL_TEXTURE0)

        SplineOpenGLFrame.CreateGLResources(self)

        glUseProgram(self.surface3Program.surfaceProgram)
        self.uSurfaceTextureMap = glGetUniformLocation(self.surface3Program.surfaceProgram, 'uTextureMap')
        glUniform1i(self.uSurfaceTextureMap, 2) # GL_TEXTURE2 is the texture map
        self.surface3Program.surfaceProgram.check_validate() # Now that textures are assigned, we can validate the program
        glUseProgram(0)

def CreateSplineFromMesh(xRange, zRange, yFunction):
    order = (3, 3)
    coefficients = np.zeros((3, xRange[2], zRange[2]), np.float32)
    knots = (np.zeros(xRange[2] + order[0], np.float32), np.zeros(zRange[2] + order[1], np.float32))
    knots[0][0] = xRange[0]
    knots[0][1:xRange[2]+1] = np.linspace(xRange[0], xRange[1], xRange[2], dtype=np.float32)[:]
    knots[0][xRange[2]+1:] = xRange[1]
    knots[1][0] = zRange[0]
    knots[1][1:zRange[2]+1] = np.linspace(zRange[0], zRange[1], zRange[2], dtype=np.float32)[:]
    knots[1][zRange[2]+1:] = zRange[1]
    for i in range(xRange[2]):
        for j in range(zRange[2]):
            coefficients[0, i, j] = knots[0][i]
            coefficients[1, i, j] = yFunction(knots[0][i], knots[1][j])
            coefficients[2, i, j] = knots[1][j]
    
    return Spline(2, 3, order, (xRange[2], zRange[2]), knots, coefficients)

if __name__=='__main__':
    viewer = Viewer(SplineOpenGLFrame=FunOpenGLFrame)
    viewer.list(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: np.sin(4*np.sqrt(x*x + y*y))))
    viewer.list(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x + y*y - 1))
    viewer.list(CreateSplineFromMesh((-1, 1, 10), (-1, 1, 8), lambda x, y: x*x - y*y))
    for i in range(16):
        viewer.list(Spline(1, 3, (3,), (5,), (np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5], np.float32),), np.array([[-1, 0, 0], [-0.5, i/16.0, 0], [0,0,0], [0.5, -i/16.0, 0], [1,0,0]], np.float32)))
    for spline in Spline.load("examples/TomsNasty.json"):
        viewer.list(spline)
    viewer.mainloop()