import numpy as np
from OpenGL.GL import *
from PIL import Image
from bspy.splineOpenGLFrame import SplineOpenGLFrame

class FunOpenGLFrame(SplineOpenGLFrame):

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
        SplineOpenGLFrame.CreateGLResources(self)

        #img = Image.open("C:/Users/ericb/OneDrive/Pictures/Backgrounds/2020 Cannon Beach.jpg")
        img = Image.open("C:/Users/ericb/OneDrive/Pictures/Backgrounds/Tom Sailing.jpg")
        img_data = np.array((img.getdata()), np.int8)
        self.textureBuffer = glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.textureBuffer)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)            
        glActiveTexture(GL_TEXTURE0)

        glUseProgram(self.surfaceProgram)
        self.uSurfaceTextureMap = glGetUniformLocation(self.surfaceProgram, 'uTextureMap')
        glUniform1i(self.uSurfaceTextureMap, 1) # GL_TEXTURE1 is the texture map
        self.surfaceProgram.check_validate() # Now that textures are assigned, we can validate the program
        glUseProgram(0)