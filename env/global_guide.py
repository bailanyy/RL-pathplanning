import math
'''
    对象Map,主要有地图数据、起点和终点
'''
class Map(object):
    def __init__(self,mapdata,startx,starty,endx,endy):
        self.data = mapdata
        self.startx = startx
        self.starty = starty
        self.endx = endx
        self.endy = endy

'''
    Node.py主要是描述对象Node
'''
class Node(object):
    '''
        初始化节点信息
    '''
    def __init__(self,x,y,g,h,father):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.father = father
    '''
        处理边界和障碍点
    '''
    def getNeighbor(self,mapdata,endx,endy):
        x = self.x
        y = self.y
        result = []
    #先判断是否在上下边界
    #if(x!=0 or x!=len(mapdata)-1):
    #上
    #Node(x,y,g,h,father)
        if(x!=0 and mapdata[x-1][y]!=1):
            upNode = Node(x-1,y,self.g+10,(abs(x-1-endx)+abs(y-endy))*10,self)
            result.append(upNode)
    #下
        if(x!=len(mapdata)-1 and mapdata[x+1][y]!=1):
            downNode = Node(x+1,y,self.g+10,(abs(x+1-endx)+abs(y-endy))*10,self)
            result.append(downNode)
    #左
        if(y!=0 and mapdata[x][y-1]!=1):
            leftNode = Node(x,y-1,self.g+10,(abs(x-endx)+abs(y-1-endy))*10,self)
            result.append(leftNode)
    #右
        if(y!=len(mapdata[0])-1 and mapdata[x][y+1]!=1):
            rightNode = Node(x,y+1,self.g+10,(abs(x-endx)+abs(y+1-endy))*10,self)
            result.append(rightNode)
    #西北  14
        if(x!=0 and y!=0 and mapdata[x-1][y-1]!=1 ):
            wnNode = Node(x-1,y-1,self.g+14,(abs(x-1-endx)+abs(y-1-endy))*10,self)
            result.append(wnNode)
    #东北
        if(x!=0 and y!=len(mapdata[0])-1 and mapdata[x-1][y+1]!=1 ):
            enNode = Node(x-1,y+1,self.g+14,(abs(x-1-endx)+abs(y+1-endy))*10,self)
            result.append(enNode)
    #西南
        if(x!=len(mapdata)-1 and y!=0 and mapdata[x+1][y-1]!=1):
            wsNode = Node(x+1,y-1,self.g+14,(abs(x+1-endx)+abs(y-1-endy))*10,self)
            result.append(wsNode)
    #东南
        if(x!=len(mapdata)-1 and y!=len(mapdata[0])-1 and mapdata[x+1][y+1]!=1 ):
            esNode = Node(x+1,y+1,self.g+14,(abs(x+1-endx)+abs(y+1-endy))*10,self)
            result.append(esNode)
        return result
    def hasNode(self,worklist):
        for i in worklist:
            if(i.x==self.x and i.y ==self.y):
                return True
        return False
    #在存在的前提下
    def changeG(self,worklist):
        for i in worklist:
            if(i.x==self.x and i.y ==self.y):
                if(i.g>self.g):
                    i.g = self.g



def getKeyforSort(element:Node):
    return element.g #element#不应该+element.h，否则会穿墙
def astar(workMap):
    startx,starty = workMap.startx,workMap.starty
    endx,endy = workMap.endx,workMap.endy
    startNode = Node(startx, starty, 0, 0, None)
    openList = []
    lockList = []
    lockList.append(startNode)
    currNode = startNode
    while((endx,endy) != (currNode.x,currNode.y)):
        workList = currNode.getNeighbor(workMap.data,endx,endy)
        for i in workList:
            if (i not in lockList):
                if(i.hasNode(openList)):
                    i.changeG(openList)
                else:
                    openList.append(i)
        openList.sort(key=getKeyforSort)#关键步骤
        currNode = openList.pop(0)
        lockList.append(currNode)
    resultx = []
    resulty = []
    while(currNode.father!=None):
        resultx.append(currNode.x)
        resulty.append(currNode.y)
        currNode = currNode.father
    resultx.append(currNode.x)
    resulty.append(currNode.y)
    return resultx,resulty