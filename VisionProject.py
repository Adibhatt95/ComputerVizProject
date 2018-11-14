import os
from scipy import misc
import numpy as np
import math
def readimg(path):
    path1 = ''
    image= misc.imread(os.path.join(path1,path), flatten= 0)

    #image = np.asarray(image)
    print(type(image))
    print(image)
    return image
image = readimg('zebra-crossing-1.bmp')
gaussMask = np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]])
print(gaussMask)
def getSubArray(arr, i,j,dist):
    starti = i-dist
    startj = j-dist
    endi = i+dist
    endj = j+dist
    retarr = np.ones([7,7],dtype=int)
    w=0
    v=0
    for x in range(starti, endi+1):
        v=0
        for y in range(startj, endj+1):
            retarr[w][v] = arr[x][y]
            v = v+1
        w = w + 1
        #print(retarr)
    return retarr

def performConv(mask,arr):
    sum = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            sum += mask[i][j] * arr[i][j]
            
   # print(sum/140)
    return sum

def gaussSmoothing(image):
    # retarr = getSubArray(image,3,3)
    gaussRes = np.zeros(image.shape, dtype=float)
    # gaussRes[3][3] = performConv(gaussMask,retarr)

    for i in range(3,image.shape[0]-3):
        for j in range(3,image.shape[1]-3):
            #if(i < 3 or i > (image.shape(0)-3) or j < 3 or j > image.shape(1)-3):
            gaussRes[i][j] = performConv(gaussMask,getSubArray(image,i,j,3))/140
    #print(retarr)
    for i in range(image.shape[1]):
        print(gaussRes[3][i])
    print(gaussRes)
    return gaussRes

gaussRes = gaussSmoothing(image)
print(gaussRes)

def normPrewittsRes(gaussRes):
    prewY = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    prewX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    maxX = 0
    maxY = 0
    normXGradient = np.zeros(gaussRes.shape, dtype=float)
    normYGradient = np.zeros(gaussRes.shape, dtype=float)
    for i in range(4,gaussRes.shape[0]-4):
        for j in range(4,gaussRes.shape[1]-4):
            normXGradient[i][j] = abs(performConv(prewX,getSubArray(gaussRes,i,j,1)))
            normYGradient[i][j] = abs(performConv(prewY,getSubArray(gaussRes,i,j,1)))
            if(abs(maxX) < abs(normXGradient[i][j])):
                maxX = normXGradient[i][j]
            if(abs(maxY) < abs(normYGradient[i][j])):
                maxY = normYGradient[i][j]
    for i in range(4,gaussRes.shape[0]-4):
        for j in range(4,gaussRes.shape[1]-4):
            normXGradient[i][j] = 255/maxX * normXGradient[i][j]
            normYGradient[i][j] = 255/maxY * normYGradient[i][j]
    return normXGradient, normYGradient

prewRes = normPrewittsRes(gaussRes)
# for i in range(prewRes[0].shape[1]):
#     print(prewRes[4][i])

def normEdgeMag(prewRes):
    edgeMagRes = np.zeros(prewRes[0].shape,dtype=float)
    max = 0
    for i in range(prewRes[0].shape[0]):
        for j in range(prewRes[0].shape[1]):
            edgeMagRes[i][j] = math.sqrt(prewRes[0][i][j] * prewRes[0][i][j] + prewRes[1][i][j] * prewRes[1][i][j])
            if(max < edgeMagRes[i][j]):
                max = edgeMagRes[i][j]
    for i in range(prewRes[0].shape[0]):
        for j in range(prewRes[0].shape[1]):
            edgeMagRes[i][j] = 255/max * edgeMagRes[i][j]
    return edgeMagRes

edgeMagRes = normEdgeMag(prewRes)
def createimages(gaussRes, prewRes, edgeMagRes):
    misc.imsave('gaussRes.bmp',gaussRes)
    misc.imsave('gradientX.bmp',prewRes[0])
    misc.imsave('gradienty.bmp',prewRes[1])
    misc.imsave('edgeMagRes.bmp',edgeMagRes)
createimages(gaussRes,prewRes,edgeMagRes)

    

