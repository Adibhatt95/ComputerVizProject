import os
from scipy import misc#for reading the image
import numpy as np#for numerical calculations on the pixel array of the image
import math#for operations like tan inverse an square root function
def readimg(path):
    path1 = ''#this shows that it is in same directory#this function takes name of the image as input, or the full path of image
    image= misc.imread(os.path.join(path1,path), flatten= 0)#this function reads images in the current version of scipy
    #it will give a warning to use the updated method to read images, but it will work because it is compatible with current version of scipy
    print(image)#tthis prints the numpy array of the pixels of the image
    return image
imagePath = 'lena256.bmp'#specify image path here.
image = readimg(imagePath)# calls the reading function


def getSubArray(arr, i,j,dist):#this function takes a numpy array arr as input, and a i,j value in the array, and a dist metric and 
    #this function returns a 7x7 sub array of the array arr which center is at i,j in arr and contains value of the arr that were distance dist in all directions
    #of the center ar i,j
    starti = i-dist#for starting the loop
    startj = j-dist
    endi = i+dist#to end the loop
    endj = j+dist
    retarr = np.ones([7,7],dtype=int)#initialize array here
    w=0
    v=0
    for x in range(starti, endi+1):
        v=0
        for y in range(startj, endj+1):
            retarr[w][v] = arr[x][y]#this fills up the new array from left to right, so values towards the right bottom corner stay the initial value
            v = v+1
        w = w + 1
        #print(retarr)
    return retarr#this is the subarray that is returned.

def performConv(mask,arr):#this function performs convolution and returns the value
    #mask=mask/operator to use
    #arr=subarray touse 
    sum = 0
    for i in range(mask.shape[0]): #this uses the dimensions of the mask as the arr dimensions will always be 7x7
        for j in range(mask.shape[1]):#the dimensions of the mask matters here.
            sum += mask[i][j] * arr[i][j]
            
   # print(sum/140)
    return sum

def gaussSmoothing(image):#this is the function that does the gaussian smoothing of image given input image.
    #image= input numpy array 
    gaussMask = np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]])
    #initialize the gaussian mask above to make it a numpy array, this is used in gaussSmoothing function
    gaussRes = np.zeros(image.shape, dtype=int)
    for i in range(3,image.shape[0]-3):
        for j in range(3,image.shape[1]-3):
            #if(i < 3 or i > (image.shape(0)-3) or j < 3 or j > image.shape(1)-3): #below, dist = 3 for gaussian 7x7 subarray
            gaussRes[i][j] = performConv(gaussMask,getSubArray(image,i,j,3))/140#uses performConv method sends mask and subarray, divide by 140 to normalize
    #print(gaussRes)
    return gaussRes

gaussRes = gaussSmoothing(image)#calls gaussian smoothing function with input image numpy array

def normPrewittsRes(gaussRes):#this function returns result of using horizontalOperator,vertical Operator
    #GAussRes= input numpy array 
    prewY = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])#initializing Prewitt operators This is for vertical Operator Gy
    prewX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])# this is for horizontal operator Gx
    normXGradient = np.zeros(gaussRes.shape, dtype=float)#initliazing the result here,
    normYGradient = np.zeros(gaussRes.shape, dtype=float)#same as above
    for i in range(4,gaussRes.shape[0]-4):#range from 4th row to 4th last row
        for j in range(4,gaussRes.shape[1]-4):#range from valid columns as above
            normXGradient[i][j] = abs(performConv(prewX,getSubArray(gaussRes,i,j,1)))/3#divide by 3 to normalize
            normYGradient[i][j] = abs(performConv(prewY,getSubArray(gaussRes,i,j,1)))/3#getting subway to convolute with the operators, dist=1 for this
    #         if(abs(maxX) < abs(normXGradient[i][j])):
    #             maxX = normXGradient[i][j]
    #         if(abs(maxY) < abs(normYGradient[i][j])):#this part was for normalization using min-max normalization, but it has been commented out.
    #             maxY = normYGradient[i][j]
    # for i in range(4,gaussRes.shape[0]-4):
    #     for j in range(4,gaussRes.shape[1]-4):
    #         normXGradient[i][j] = 255/maxX * normXGradient[i][j]
    #         normYGradient[i][j] = 255/maxY * normYGradient[i][j]
    return normXGradient, normYGradient
prewRes = normPrewittsRes(gaussRes)#calling the function


def normEdgeMag(prewRes):#calculates edge magnitudes and normalizes and returns the result.
    edgeMagRes = np.zeros(prewRes[0].shape,dtype=float)#result initialized
    max = 0.0
    for i in range(prewRes[0].shape[0]):
        for j in range(prewRes[0].shape[1]):
            edgeMagRes[i][j] = math.sqrt(prewRes[0][i][j] * prewRes[0][i][j] + prewRes[1][i][j] * prewRes[1][i][j])/math.sqrt(2)#calculation here
            #dividing by math.sqrt(2) for normalization
            # if(max < edgeMagRes[i][j]):
            #     max = edgeMagRes[i][j]
    # for i in range(prewRes[0].shape[0]):
    #     for j in range(prewRes[0].shape[1]):
    #         edgeMagRes[i][j] = 255/max * edgeMagRes[i][j]
    # print('max:{}'.format(max))
    return edgeMagRes

edgeMagRes = normEdgeMag(prewRes)
def createimage(Res, name, imagePath): #function to write images.
    #Res = numpy array to write image
    #name= name of image
    #imagepath = original image name to front_append name to 
    imagePath = imagePath.split('.')[0]
    imagePath += '_'+name+'.bmp'
    misc.imsave(imagePath,Res)

def getSection(angle):#function to get the correct section from the gradient angle.
    #angle=gradient angle
    #this checks for all 4 sections, since inverse of tan's range is -90 to +90, the angles outside this range will not be encountered ever.
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

def getMagWithAngle(edgeMagRes,section,i,j): #Function that checks the magnitudes of neighbours in that section
    #edgeMagRes =numpy array, section= section based on the angle, i and j- coordinates of pixel in edgeMagRes to compare magnitude with.
    if section == 1:
        if(edgeMagRes[i][j] >= edgeMagRes[i][j-1] and  edgeMagRes[i][j] >= edgeMagRes[i][j+1]):
            return math.floor(edgeMagRes[i][j]) #this function is used to convert decmial float to integer, different functions for this will give difffernt answers
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
    edgesCount = 0
    thesholdRes = np.zeros(nonMaxSuppRes.shape,dtype=int)
    for i in range(nonMaxSuppRes.shape[0]):
        for j in range(nonMaxSuppRes.shape[1]):
            if nonMaxSuppRes[i][j] == 0:
                thesholdRes[i][j] = 0
            elif nonMaxSuppRes[i][j] >= threshold:
                thesholdRes[i][j] = 255
                edgesCount+=1
            else:
                thesholdRes[i][j] = 0
    return thesholdRes,edgesCount

threshold10 = getPTileThreshold(10,nonMaxSuppRes)
threshold30 = getPTileThreshold(30,nonMaxSuppRes)
threshold50 = getPTileThreshold(50,nonMaxSuppRes)
thrs10Res = getResAfterThreshold(threshold10,nonMaxSuppRes)
thrs30Res = getResAfterThreshold(threshold30,nonMaxSuppRes)
thrs50Res = getResAfterThreshold(threshold50,nonMaxSuppRes)
createimage(thrs10Res[0], '10PercentThreshold',imagePath)
createimage(thrs30Res[0], '30PercentThreshold',imagePath)
createimage(thrs50Res[0], '50PercentThreshold',imagePath)   
print('{} {} {} are the thresholds for 10%, 30%, and 50percent respectively'.format(threshold10,threshold30,threshold50))
print('{} {} {} are the total number of edges detected for 10%, 30%, and 50percent ptile respectively'.format(thrs10Res[1],thrs30Res[1],thrs50Res[1]))

