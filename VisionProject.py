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
imagePath = 'lena256.bmp'
image = readimg(imagePath)
#image = readimg('gradientX.bmp')
for i in range(0,5):
    print(i)
gaussMask = np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]])
# print(gaussMask)
# np.savetxt('file.txt', image, fmt="%d")
# image2 = readimg('zebra-crossing-1_30%.bmp')
# image3 = readimg('zebra-crossing-1_nonMaximaSuppressionResult.bmp')
# image4 = readimg('')
# np.savetxt('file2.txt', image2, fmt="%d")
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         if image[i][j] != image2[i][j]:
#             print('{} {} {} {}'.format(i,j,image[i][j],image2[i][j]))
# import pdb
# pdb.set_trace()

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
    gaussRes = np.zeros(image.shape, dtype=int)
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
# #print(gaussRes)
# import pdb
# pdb.settrace()
def normPrewittsRes(gaussRes):
    prewY = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    prewX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    maxX = 0
    maxY = 0
    normXGradient = np.zeros(gaussRes.shape, dtype=float)
    normYGradient = np.zeros(gaussRes.shape, dtype=float)
    for i in range(4,gaussRes.shape[0]-4):
        for j in range(4,gaussRes.shape[1]-4):
            normXGradient[i][j] = abs(performConv(prewX,getSubArray(gaussRes,i,j,1)))/3
            normYGradient[i][j] = abs(performConv(prewY,getSubArray(gaussRes,i,j,1)))/3
    #         if(abs(maxX) < abs(normXGradient[i][j])):
    #             maxX = normXGradient[i][j]
    #         if(abs(maxY) < abs(normYGradient[i][j])):
    #             maxY = normYGradient[i][j]
    # for i in range(4,gaussRes.shape[0]-4):
    #     for j in range(4,gaussRes.shape[1]-4):
    #         normXGradient[i][j] = 255/maxX * normXGradient[i][j]
    #         normYGradient[i][j] = 255/maxY * normYGradient[i][j]
    return normXGradient, normYGradient

prewRes = normPrewittsRes(gaussRes)
# for i in range(prewRes[0].shape[1]):
#     print(prewRes[4][i])

def normEdgeMag(prewRes):
    edgeMagRes = np.zeros(prewRes[0].shape,dtype=float)
    max = 0.0
    for i in range(prewRes[0].shape[0]):
        for j in range(prewRes[0].shape[1]):
            edgeMagRes[i][j] = math.sqrt(prewRes[0][i][j] * prewRes[0][i][j] + prewRes[1][i][j] * prewRes[1][i][j])/math.sqrt(2)
            if(max < edgeMagRes[i][j]):
                max = edgeMagRes[i][j]
    # for i in range(prewRes[0].shape[0]):
    #     for j in range(prewRes[0].shape[1]):
    #         edgeMagRes[i][j] = 255/max * edgeMagRes[i][j]
    print('max:{}'.format(max))
    return edgeMagRes

edgeMagRes = normEdgeMag(prewRes)
def createimage(Res, name, imagePath):
    imagePath = imagePath.split('.')[0]
    imagePath += '_'+name+'.bmp'
    misc.imsave(imagePath,Res)

def getSection(angle):
    if angle < 0:
        angle = 360 + angle
    if (337.5 <= angle and angle <= 360) or (0 <=  angle and angle < 22.5):
        return 1
    elif 67.5 > angle and angle >= 22.5:
        return 2
    elif (67.5 <= angle and angle < 112.5) or (247.5 <= angle and angle < 292.5):
        return 3
    elif (292.5 <= angle and angle < 337.5):
        return 4
    else:
        print('UNNKWWONAOWNOWAONAWONOWANAOWNDUABFKJHKJFHKHKJAWBKHAJLFHJK:ADSFNWMFJNFHJDKGH:WJNMFOIBGFHJKHFGKJHDLKFNLJKWHFLHLAKEFNLFDNss')
        return 'unknown {}'.format(angle)

def getMagWithAngle(edgeMagRes,section,i,j): #need to check for greater than or equal to thing here 
    if section == 1:
        if(edgeMagRes[i][j] >= edgeMagRes[i][j-1] and  edgeMagRes[i][j] >= edgeMagRes[i][j+1]):
            return math.floor(edgeMagRes[i][j])
        else:
            return 0
    elif section == 2:
        if(edgeMagRes[i][j] >= edgeMagRes[i+1][j-1] and  edgeMagRes[i][j] >= edgeMagRes[i-1][j+1]):
            return math.floor(edgeMagRes[i][j])
        else:
            return 0
    elif section == 3:
        if(edgeMagRes[i][j] >= edgeMagRes[i+1][j] and  edgeMagRes[i][j] >= edgeMagRes[i-1][j]):
            return math.floor(edgeMagRes[i][j])
        else:
            return 0
    elif section == 4:
        if(edgeMagRes[i][j] >= edgeMagRes[i-1][j-1] and  edgeMagRes[i][j] >= edgeMagRes[i+1][j+1]):
            return math.floor(edgeMagRes[i][j])
        else:
            return 0
    else:
        return 'disaster {} {} {}'.format(section,i,j)

def getAngle(y,x):
    if x == 0 and y != 0:
        return 90
    elif(y == 0):
        return 0
    else:
        return math.degrees(math.atan2(y,x))

def nonMaxSupp(edgeMagRes, prewRes):
    nonMaxSuppRes = np.zeros(edgeMagRes.shape,dtype=int)
    for i in range(5,prewRes[0].shape[0]-5):
        for j in range(5,prewRes[0].shape[1]-5):
            angle = getAngle(prewRes[1][i][j],prewRes[0][i][j])
            #pdb.set_trace()
            section = getSection(angle)
            nonMaxSuppRes[i][j] = getMagWithAngle(edgeMagRes,section,i,j)
            #print(nonMaxSuppRes[i][j])
    return nonMaxSuppRes

nonMaxSuppRes = nonMaxSupp(edgeMagRes,prewRes)
createimage(gaussRes,'GaussRes',imagePath)
createimage(prewRes[0],'PrewittXResult',imagePath)
createimage(prewRes[1],'PrewittYResult',imagePath)
createimage(edgeMagRes,'EdgeMagnitudeResult',imagePath)
createimage(nonMaxSuppRes,'nonMaximaSuppressionResult',imagePath)
def getPTileThreshold(percent, nonMaxSuppRes):
    dictImg = {}
    #import pdb
    #pdb.set_trace()
    totalTest = nonMaxSuppRes.shape[0] * nonMaxSuppRes.shape[1]
    for i in range(nonMaxSuppRes.shape[0]):
        for j in range(nonMaxSuppRes.shape[1]):
            if nonMaxSuppRes[i][j] == 0:
                totalTest = totalTest-1
                continue
            elif nonMaxSuppRes[i][j] in dictImg:
                dictImg[nonMaxSuppRes[i][j]] += 1
            else:
                dictImg[nonMaxSuppRes[i][j]] = 1
    totalPixels = 0
    for i in range(1, 256):
        if i in dictImg:
            totalPixels += dictImg[i]
    topPixels = int(totalPixels * percent/100)
    threshold = 0
    print('total:{} {}'.format(totalPixels,totalTest))
    #pdb.set_trace()
    for i in range(0,256):
        if i in dictImg:
            topPixels = topPixels - dictImg[i]
        if topPixels <= 0:
            threshold = i
            #pdb.set_trace()
            break
    print(threshold)
    topPixels = int(totalPixels * percent/100)
    threshold = 0
    for i in range(255,-1,-1):
        if i in dictImg:
            topPixels = topPixels - dictImg[i]
        if topPixels < 0:
            threshold = i
            # import pdb
            # pdb.set_trace()
            break
        elif topPixels == 0:
            threshold = i-1
    return threshold
def getResAfterThreshold(threshold,nonMaxSuppRes):
    thesholdRes = np.zeros(nonMaxSuppRes.shape,dtype=int)
    for i in range(nonMaxSuppRes.shape[0]):
        for j in range(nonMaxSuppRes.shape[1]):
            if nonMaxSuppRes[i][j] == 0:
                thesholdRes[i][j] = 0
            elif nonMaxSuppRes[i][j] >= threshold:
                thesholdRes[i][j] = 255
            else:
                thesholdRes[i][j] = 0
    return thesholdRes

threshold10 = getPTileThreshold(10,nonMaxSuppRes)
threshold30 = getPTileThreshold(30,nonMaxSuppRes)
threshold50 = getPTileThreshold(50,nonMaxSuppRes)
thrs10Res = getResAfterThreshold(threshold10,nonMaxSuppRes)
createimage(thrs10Res, '10PercentThreshold',imagePath)
print('{} {} {} this is the one'.format(threshold10,threshold30,threshold50))

