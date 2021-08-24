# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:23:20 2021

@author: tunah
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
#from grid_map_lib.grid_map_lib import GridMap

class Point:
    def __init__(self, x ,y):
        self.x = x
        self.y = y

class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2



INT_MAX = 10000
 

def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 

def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
 
def doIntersect(p1, q1, p2, q2):
     
    # Find the four orientations needed for 
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # Special Cases
    # p1, q1 and p2 are colinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are colinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are colinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are colinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 
# Returns true if the point p lies 
# inside the polygon[] with n vertices
def is_inside_polygon(points:list, p:tuple) -> bool:
     
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):
                             
            
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
         
    
    return (count % 2 == 1)


def isSlopeInfinity(point1, point2):
    return point2[0] == point1[0]
        

def intersection(neededinfo1, neededinfo2):
    
    if(neededinfo1[1] == 0 and neededinfo2[2] == 0):
        return [neededinfo2[3], neededinfo1[3]]
    elif(neededinfo2[1] == 0 and neededinfo1[2] == 0):
        return [neededinfo1[3], neededinfo2[3]]
    elif(neededinfo1[1] == 0):
        
        y = neededinfo1[3]
        x = (-1*y*neededinfo2[2] - neededinfo2[3])/ neededinfo2[1]
        return [x, y]
    elif(neededinfo2[1] == 0):
        
        y = neededinfo2[3]
        x = (-1*y*neededinfo1[2] - neededinfo1[3])/ neededinfo1[1]
        return [x, y]
    elif(neededinfo1[2] == 0):
        
        x = neededinfo1[3]
        y = (-1*x*neededinfo1[2] - neededinfo2[3])/ neededinfo2[1]
        return [x, y]    
    elif(neededinfo2[2] == 0):
        
        x = neededinfo2[3]
        y = (-1*x*neededinfo2[2] - neededinfo1[3])/ neededinfo1[1]
        return [x, y]   
    else:
        
        y = (neededinfo1[1]*neededinfo2[3] - neededinfo2[1]*neededinfo1[3]) / (neededinfo2[1]*neededinfo1[2] - neededinfo1[1]*neededinfo2[2])
        x = (-1*y*neededinfo1[2] - neededinfo1[3]) / (neededinfo1[1])
        return [x,y]
    
    
def farthestForInf(point1, index1):
    maxx = -1
    indexx = -1
    for x in range(len(newlistOfCoors[0]) - 1):
        if(x != index1):
            tmp = abs(point1[0] - newlistOfCoors[0][x])
            if(tmp > maxx):
                maxx = tmp
                indexx = x
    return indexx
        
def slopeaxbyc(point1, point2):
    
    slope = float(point2[1] - point1[1]) / (point2[0] - point1[0])
    ax = slope
    by = -1
    c = -1.0*slope*point2[0] + point2[1]
    
    return [slope, ax, by, c, point1, point2]
            
def findFarthest(info ,index1, index2):
    maxxd = -1
    farthest = -1
    for k in range(len(newlistOfCoors[0]) - 1):
                
        if(k!=index1 and k!=index2):
                    
            dist = abs(info[1]*newlistOfCoors[0][k] + info[2]*newlistOfCoors[1][k] + info[3]) / (math.sqrt(info[1]*info[1] + info[2]*info[2]))
                    
            if(dist > maxxd):
                        
                maxxd = dist
                farthest = k
    return farthest
            
def findStoppingc(reachP, info):
    
      return (-1.0*info[1]*reachP[0] -info[2]*reachP[1] )

global listOfCoors
listOfCoors = []

global newlistOfCoors
newlistOfCoors = [[], []]

def onClick(event):
    """determines the colour and marker of drones and targets for manual drone and target placement initiator"""
    fig = plt.gcf()
    plt.plot(event.xdata, event.ydata, 'o', color='b')
    listOfCoors.append([event.xdata, event.ydata])
    fig.canvas.draw()
        
def onKey(event):
    if event.key == 'enter':
        
        whichCorners = []
        maxxLen = -1
        for k in range(len(listOfCoors)):
            if (math.sqrt( (listOfCoors[k][0]- listOfCoors[(k+1)%len(listOfCoors)][0])**2 +(listOfCoors[k][1]- listOfCoors[(k+1)%len(listOfCoors)][1])**2  ) > maxxLen):
                whichCorners = [k, (k+1)%len(listOfCoors)]
                maxxLen = math.sqrt( (listOfCoors[k][0]- listOfCoors[(k+1)%len(listOfCoors)][0])**2 +(listOfCoors[k][1]- listOfCoors[(k+1)%len(listOfCoors)][1])**2  )
        
        isSlopeInf = False
        
        if((listOfCoors[whichCorners[1]][0] -  listOfCoors[whichCorners[0]][0]) == 0):
            isSlopeInf = True
        
        ax = 1
        by = 0
        c = -1.0*listOfCoors[whichCorners[0]][0]
        slope = -1
        
        if not isSlopeInf:
            
            slope = (float(listOfCoors[whichCorners[1]][1] -  listOfCoors[whichCorners[0]][1])) / (listOfCoors[whichCorners[1]][0] -  listOfCoors[whichCorners[0]][0])
            ax = slope
            by = -1
            c = -1.0*slope*listOfCoors[whichCorners[1]][0] + listOfCoors[whichCorners[1]][1]
        
        
        maxxDist = -1
        farthestIndex = -1
        
        for k in range(len(listOfCoors)):
        
            if((k!=whichCorners[0] and k!=whichCorners[1])):
                
                dist = abs(ax*listOfCoors[k][0] + by*listOfCoors[k][1] + c) / (math.sqrt(ax*ax + by*by))
                
                if(dist > maxxDist):
                    
                    maxxDist = dist
                    farthestIndex = k
        stoppingC = -1.0*ax*listOfCoors[farthestIndex][0] - by*listOfCoors[farthestIndex][1]
        listOfCoors.append(listOfCoors[0])
        tempx = []
        tempy = []
        for elemx in listOfCoors:
            tempx.append(elemx[0])
            tempy.append(elemx[1])
        
        newlistOfCoors[0] = tempx
        newlistOfCoors[1] = tempy 
        
        toAdd = math.sqrt(ax*ax + by*by)
        
        plt.plot(tempx, tempy , label = 'Scanning Area', marker = 'o')
        
        firstLong = []
        
        backFirstLong = []
        InformationOfTheLongestLines = []
        
        if not isSlopeInf:
            
            if(stoppingC  > c ):
                tempc = c + toAdd 
                firstLong = [slope, ax, by, c + 0.999999999*toAdd]
                backFirstLong = [slope, ax, by, c+ 0.333*toAdd]
                while (tempc < stoppingC):
                    InformationOfTheLongestLines.append([slope, ax, by, tempc])
                    tempc = tempc + toAdd
            else:
                 
                 tempc = c - toAdd 
                 firstLong = [slope, ax, by, c - 0.999999999*toAdd]
                 backFirstLong = [slope, ax, by, c - 0.333*toAdd]
                 while (tempc > stoppingC):
                    InformationOfTheLongestLines.append([slope, ax, by, tempc])
                    
                    
                    tempc = tempc - toAdd
            
        else:
            
            reachPoint = newlistOfCoors[0][farthestIndex]
            whereWeare = newlistOfCoors[0][whichCorners[0]]
            if(reachPoint > whereWeare ):
                whereWeare = whereWeare + toAdd
                backFirstLong = [slope, ax, by, c+ 0.333*toAdd]
                firstLong = [slope, ax, by, c+ 0.9999*toAdd]
                while(reachPoint > whereWeare):
                    
                    InformationOfTheLongestLines.append([slope, ax, by, whereWeare])
                    whereWeare = whereWeare + toAdd
            else:
                whereWeare = whereWeare - toAdd
                backFirstLong = [slope, ax, by, c - 0.333*toAdd]
                firstLong = [slope, ax, by, c+ 0.99999*toAdd]
                while(reachPoint < whereWeare):
                    
                    InformationOfTheLongestLines.append([slope, ax, by, whereWeare])
                    whereWeare = whereWeare - toAdd
        
        
        #time to find the windows
        
        
        listOfWindows = []
        listOfInside = []
        listOfBetween = []
        gottagoback = []
        for k in range(len(newlistOfCoors[0]) - 1):
            ll = len(newlistOfCoors[0]) - 1
            if(k!=whichCorners[0]):
                
                if(isSlopeInfinity([newlistOfCoors[0][k], newlistOfCoors[1][k]],  [newlistOfCoors[0][(k+1)%ll],  newlistOfCoors[1][(k+1)%ll] ] ) == True):
                    
                    indexx = farthestForInf([ newlistOfCoors[0][k], newlistOfCoors[1][k]], k)
                    
                    if(newlistOfCoors[0][indexx] > newlistOfCoors[0][k]):
                         gottagoback.append([-1, 1 ,0, newlistOfCoors[0][k] + 0.333 ])
                         listOfWindows.append([-1, 1 ,0, newlistOfCoors[0][k] + 1 ])
                         listOfInside.append([-1, 1 ,0, newlistOfCoors[0][k] + 1 ])
                         listOfBetween.append([-1, 1 ,0, newlistOfCoors[0][k] + 0.9999999999 ])
                    else:
                         gottagoback.append([-1, 1 ,0, newlistOfCoors[0][k] - 0.333 ])
                         listOfWindows.append([-1, 1 ,0, -1*newlistOfCoors[0][k] - 1])
                         listOfInside.append([-1, 1 ,0, newlistOfCoors[0][k] - 1 ])
                         listOfBetween.append([-1, 1 ,0, newlistOfCoors[0][k] - 0.999999999 ])
                        
                        
                    
                    
                else:
                    
                    neededInfo = slopeaxbyc([newlistOfCoors[0][k], newlistOfCoors[1][k]],  [newlistOfCoors[0][(k+1)%ll],  newlistOfCoors[1][(k+1)%ll]])
                    farthest = findFarthest(neededInfo, k, (k+1) % ll)
                    stopc = findStoppingc([newlistOfCoors[0][farthest], newlistOfCoors[1][farthest]], neededInfo)
                    zz = math.sqrt(neededInfo[1]**2 + neededInfo[2]**2)
                    tmp = copy.deepcopy(neededInfo)
                    tmp2 = copy.deepcopy(neededInfo)
                    tmp3 = copy.deepcopy(neededInfo)
                    if(stopc > neededInfo[3]):
                         
                         
                         t1 = neededInfo[3] + float(zz)
                         t2 = neededInfo[3] + float(zz)*0.9999999999
                         t3 = neededInfo[3] + zz*0.333
                         
                         
                         tmp[3] = t1
                         tmp2[3] = t2
                         tmp3[3] = t3
                         
                         listOfBetween.append(tmp2)
                         listOfWindows.append(tmp)
                         listOfInside.append(tmp)
                         
                         gottagoback.append(tmp3)
                         
                    else :
                        
                         tmp3[3] = neededInfo[3] - zz*0.333
                         tmp[3] = neededInfo[3] - float(zz)
                         tmp2[3] = neededInfo[3] - float(zz)*0.9999999999999
                         
                         listOfBetween.append(tmp2)
                         listOfWindows.append(tmp)
                         listOfInside.append(tmp)
                         
                         gottagoback.append(tmp3)
            else:
                
                gottagoback.append(backFirstLong)
                listOfBetween.append(firstLong)
                listOfInside.append(InformationOfTheLongestLines[0])
                
        
        linesCoveringObject = []
        
        
        for x in range(len(listOfCoors) - 1):
            if(isSlopeInfinity(listOfCoors[x], listOfCoors[x + 1])):
                linesCoveringObject.append([-1 , -1 , 0, -1*listOfCoors[x][0]])
            
            else:
                myInfo = slopeaxbyc(listOfCoors[x], listOfCoors[x + 1])
                linesCoveringObject.append(myInfo)
        
        listOfIntersectionPoints = []
        insidePolygonCoors = []
        inside2 = []
        backroadcoors = []
        
        lll = len(listOfBetween)
        
        for k in range(len(listOfBetween)):
            
            backroadcoors.append(intersection(gottagoback[k] ,gottagoback[(k+1)%lll] ))
            insidePolygonCoors.append(intersection(listOfBetween[k] ,listOfBetween[(k+1)%lll] ))
            inside2.append(intersection(listOfInside[k] ,listOfInside[(k+1)%lll] ))
        
        lastVisited = [-1,-1]
        l = len(listOfWindows)
        
        for elemLong in InformationOfTheLongestLines:
            
            currentPos = []
            elems = []
                
            for x in range(len(listOfWindows)):
                if((elemLong[1] == 0 and listOfWindows[x][1] == 0) or (elemLong[2] == 0 and listOfWindows[x][2] == 0) or (abs(elemLong[0] - listOfWindows[x][0]) < 0.000001) ):
                    continue
                    
                else:
                    intersectionPoint = intersection(elemLong, listOfWindows[x])
                    
                    
                    if(is_inside_polygon(insidePolygonCoors, intersectionPoint)):
                            
                            currentPos.append(x)
                            elems.append(intersectionPoint)
            

            
            if(len(currentPos) == 0):
                break
            
            elif(lastVisited[0] == -1):
                listOfIntersectionPoints.append(elems[0])
                listOfIntersectionPoints.append(elems[1])
                lastVisited[0] = currentPos[0]
                lastVisited[1] = currentPos[1]
                
            else:
                
                if(currentPos[0] == lastVisited[0] and currentPos[1] == lastVisited[1]):
                    
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    list.reverse(lastVisited)
                
                    
                elif(currentPos[0] == lastVisited[1] and currentPos[1] == lastVisited[0]):
                    
                    listOfIntersectionPoints.append(elems[0])
                    listOfIntersectionPoints.append(elems[1])
                    list.reverse(lastVisited)
                    
                elif(lastVisited[0] == currentPos[0]):
                  
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                    
                elif(lastVisited[1] == currentPos[1]):
                   
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                    
                elif(lastVisited[0] == currentPos[1]):
                    
                    listOfIntersectionPoints.append(elems[0])
                    listOfIntersectionPoints.append(elems[1])
                    lastVisited[0] = currentPos[0]
                    lastVisited[1] = currentPos[1]
                    
                elif(lastVisited[1] == currentPos[0]):
                    
                    listOfIntersectionPoints.append(elems[0])
                    listOfIntersectionPoints.append(elems[1])
                    lastVisited[0] = currentPos[0]
                    lastVisited[1] = currentPos[1]
                    
                elif(lastVisited[0] == l - 1 and currentPos[0] == 0 ):
                    
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                elif(lastVisited[1] == l - 1 and currentPos[0] == 0 ):
                    
                    listOfIntersectionPoints.append(elems[0])
                    listOfIntersectionPoints.append(elems[1])
                    lastVisited[0] = currentPos[0]
                    lastVisited[1] = currentPos[1]
                elif(currentPos[1] == l - 1 and lastVisited[1] == 0 ):
                    
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                    
                elif(currentPos[1] - lastVisited[0] == 1 and currentPos[0] - lastVisited[1] == -1):
                    listOfIntersectionPoints.append(elems[0])
                    listOfIntersectionPoints.append(elems[1])
                    lastVisited[0] = currentPos[0]
                    lastVisited[1] = currentPos[1]
                    
                elif(lastVisited[0] == 0 and currentPos[1] == l - 1 ):
                   
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                    
                elif(currentPos[0] - lastVisited[0] == -1 and currentPos[1] - lastVisited[1] == 1):
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                
                elif(currentPos[0] - lastVisited[0] == 1 and currentPos[1] - lastVisited[1] == -1):
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                    
                elif(currentPos[0] - lastVisited[0] == 1 and currentPos[1] - lastVisited[1] == 1):
                    listOfIntersectionPoints.append(elems[1])
                    listOfIntersectionPoints.append(elems[0])
                    lastVisited[0] = currentPos[1]
                    lastVisited[1] = currentPos[0]
                    
                elif(currentPos[0] - lastVisited[1] == 1 and  lastVisited[0] - currentPos[1] == 1):
                    listOfIntersectionPoints.append(elems[0])
                    listOfIntersectionPoints.append(elems[1])
                    lastVisited[0] = currentPos[0]
                    lastVisited[1] = currentPos[1]   
                    
                elif(currentPos[1] - lastVisited[0] == 1 and currentPos[0] - lastVisited[1] == 1):
                    listOfIntersectionPoints.append(elems[0])
                    listOfIntersectionPoints.append(elems[1])
                    lastVisited[0] = currentPos[0]
                    lastVisited[1] = currentPos[1]
                
                else:
                    print("bug")
 
                      
        tpx = []
        tpy = []
        
        
        for elem in listOfIntersectionPoints:
            tpx.append(elem[0])
            tpy.append(elem[1])
            
        
        # leng = len(insidePolygonCoors)
        
        # tf = (farthestIndex - 1 + leng)%leng
        # tpx.append(backroadcoors[tf][0])
        # tpy.append(backroadcoors[tf][1])
        
        # newX = []
        # newY = []
        # while tf != whichCorners[0]:
        #     newX.append(backroadcoors[tf][0])
        #     newY.append(backroadcoors[tf][1])
        #     tf = (tf+1)%leng   
        
        # if(whichCorners[0] == 0 or whichCorners[1] == 0):
            
        #     newX.append(backroadcoors[tf][0])
        #     newY.append(backroadcoors[tf][1])
        
        
        listOfLines = []
        
        for i in range(www):
            listOfLines.append([[], []])
            
            
        acc = 0
        inc = int(len(listOfIntersectionPoints)/www)
        
        for i in range(www):
            # print("i = "+str(i))
            # strr = "Starting point of the drone "+ str(i + 1)
            # strrr = "Shortest Coverage Path for Drone " + str(i + 1)
            if(i == 0):
                
                listOfLines[i][0] = tpx[acc:(acc + inc)]
                listOfLines[i][1] = tpy[acc: (acc + inc)]
                # plt.plot(tpx[acc], tpy[acc],marker =  "x", label = strr)
                # plt.plot(tpx[acc:(acc + inc)], tpy[acc: (acc + inc)], "--", label = strrr) 
            elif(i == www - 1):
                
                listOfLines[i][0] = tpx[acc - 1:]
                listOfLines[i][1] = tpy[acc - 1:]
                # plt.plot(tpx[acc - 1], tpy[acc - 1],marker =  "x", label =strr)
                # plt.plot(tpx[acc - 1:], tpy[acc - 1:], "--", label = strrr) 
            
            else:
                listOfLines[i][0] = tpx[(acc - 1):(acc + inc)]
                listOfLines[i][1] = tpy[(acc- 1): (acc + inc)]
                # plt.plot(tpx[acc - 1], tpy[acc - 1],marker =  "x", label = strr)
                # plt.plot(tpx[(acc - 1):(acc + inc)], tpy[(acc- 1): (acc + inc)], "--", label = strrr) 
                
            acc = acc + inc
        
        # plt.plot(tpx[len(tpx) - 1], tpy[len(tpy) - 1],marker =  "x", label = "End point of the path")
        # plt.plot(newX, newY, color = "black", label = "Return path")
        
        
        def init():
            for line in lines:
                line.set_data([],[])
            return lines
        
        print(listOfLines[0])
        def animation_frame(i):
            
            for lnum,line in enumerate(lines):
                
                line.set_data(listOfLines[lnum][0][:i], listOfLines[lnum][1][:i])
                
            return lines
        
        
        animationn= FuncAnimation(fig, animation_frame, init_func = init, frames = np.arange(1,len(tpx),1), interval = 250, blit=True)
        
        
        plt.legend()
        plt.show()
        plt.savefig("output.png")
        
    else:
        
        exit(0)
    
    
global www
www = int(input("how many drones? "))
if(www < 1):
    print("impossible")
else:
    
    
    
    fig = plt.figure() 
    ax = plt.axes(xlim=(0, 100), ylim=(0, 100)) 
    lines = []
    
    for index in range(www):
        lobj = ax.plot([],[])[0]
        lines.append(lobj)
    
    fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('key_release_event', onKey)
