import os
from time import sleep
from time import time
import random
import json

def posStr(x: int,y: int):
	return "(%d,%d)"%(x,y)


def QPosStr(x: int,y: int, action: str):
	return "(%d,%d)%s"%(x, y, action)


class QTable:
	def __init__(self, width: int, height: int, actions: list,alpha = 0.5, gamma = 0.8):
		self.height = height
		self.width = width
		
		self.Qtable = {}
		for i in range(height):
			for j in range(width):
				for act in actions:
					tempKey = QPosStr(j, i, act)
					randomV = round(random.gauss(1,1),2)
					self.Qtable[tempKey] = randomV
		
		self.fileName = "mapPos%dx%d.txt"%(self.width,self.height)
		
		self.alpha = alpha
		self.gamma = gamma
		
		self.loadFile()
		
	def readFile(self):
		f = open(self.fileName,"r")
		c = f.read()
		self.Qtable = json.loads(c)
		f.close()
		
	def saveTable(self):
		f = open(self.fileName,"w")
		c = json.dumps(self.Qtable)
		f.write(c)
		f.close()
	
	def loadFile(self):
		if os.path.exists(self.fileName):
			self.readFile()
		else:
			self.saveTable()


class Map:
	"""
	定义一个地图类，囊括了陷阱、终点和玩家。
		self.map: 根据构建的地图大小生成的一个字典，键是"(x, y)",值是"E","P","X"。
		"E": 表示终点
		"P": 表示玩家
		"X": 表示陷阱
		" "：表示正常地带
	"""
	def __init__(self, width: int, height: int):
		self.height = height
		self.width = width
		self.map = {posStr(j,i):" " for i in range(self.height) for j in range(self.width)}
		
		#addPlayer
		self.addPlayer(0,3)
		
		#addTrap
		self.addTrap(1,3)
		self.addTrap(2,3)
		self.addTrap(2,1)
		self.addTrap(2,2)
		self.addTrap(4,5)
		self.addTrap(5,4)
		
		#
		#"X" = "X"
		#"P" = "P"
		
		#addEnd
		self.addEnd(5,5)
		
		#direction
		self.directions = {
			"up":   (0,-1),
			"down": (0,1),
			"left": (-1,0),
			"right":(1,0),
			}
		
		#actions
		self.actions = ["up","down","left","right",""]
		
		#reward
		self.rewards = {
			" ":-1,
			"E":100,
			"X":-100,
			}
		
		self.currentReward = 0
		
		#legal move char
		self.legalList = [" ","E","X"]
		
		self.qtable = QTable(width, height, self.actions)
		
		self.win = False
		self.over = False
		self.currentState = "" #记录当前状态
	
	def saveQTable(self):
		"""
		保存Q表
		"""
		self.qtable.saveTable()
	
	def mapPrint(self):
		"""
		绘制地图、玩家、陷阱、终点，逐行打印。
		"""
		for i in range(self.height):
			print("*-"*self.width+"*")
			line = "|"
			for j in range(self.width):
				tempP = posStr(j,i)
				liver = self.map[tempP]
				line += liver+"|"
			print(line)
		print("*-"*self.width+"*")
		
	def isTrap(self, x: int, y: int):
		"""
		return bool
		判断当前位置是否有陷阱
		"""
		tempP = posStr(x,y)
		
		if self.map[tempP] == "X":
			return True
			
		return False
		
	def isEnd(self, x: int, y: int):
		"""
		return bool
		判断当前位置是否有陷阱
		"""
		tempP = posStr(x,y)
		
		if self.map[tempP] == "E":
			return True
			
		return False
		
	def isPlayer(self, x: int, y: int):
		"""
		return bool
		判断当前位置是否有玩家
		"""
		tempP = posStr(x,y)
		
		if self.map[tempP] == "P":
			return True
			
		return False
		
	def addPlayer(self,x: int, y: int):
		"""
		根据坐标(x, y)添加玩家
		"""
		assert x < self.width and x >= 0 ,"addPlayer is not in width [0,%d]"%self.width
		assert y < self.height and y >= 0,"addPlayer is not in height [0,%d]"%self.height
		
		tempP = posStr(x,y)
		self.map[tempP] = "P"
	
	def addTrap(self,x: int, y: int):
		"""
		根据坐标(x, y)添加陷阱
		"""
		assert x < self.width and x >= 0 ,"addTrap is not in width [0,%d]"%self.width
		assert y < self.height and y >= 0,"addTrap is not in height [0,%d]"%self.height
		
		tempP = posStr(x,y)
		self.map[tempP] = "X"
		
	def addEnd(self,x: int, y: int):
		"""
		根据坐标(x, y)添加终点
		"""
		assert x < self.width and x >= 0 ,"addEnd is not in width [0,%d]"%self.width
		assert y < self.height and y >= 0,"addEnd is not in height [0,%d]"%self.height
		
		tempP = posStr(x,y)
		self.map[tempP] = "E"
	
	def getPlayerPos(self):
		"""
		return 玩家坐标(x, y)
		"""
		for i in range(self.height):
			for j in range(self.width):
				tempP = posStr(j,i)
				liver = self.map[tempP]
				if liver == "P":
					return j, i
				
	def legalMove(self,x: int,y: int) -> bool:
		"""
		合法移动判断，不出界皆合法。
		return bool
		"""
		#in map
		if x < self.width and x >= 0 and y < self.height and y >= 0:
			
			#legal move
			if self.map[posStr(x,y)] in self.legalList:
				
				return True
				
		return False
		
	
	def gameWin(self, nextPosX: int, nextPosY: int) -> bool:
		"""
		游戏胜利判定
		"""
		tempPos = posStr(nextPosX, nextPosY)
		if self.map[tempPos] == "E":
			return True
			
		return False
	
	def gameWinUI(self, x, y):
		"""
		到达终点胜利界面
		"""
		#delete orignal player
		playerPosX, playerPosY = self.getPlayerPos()
		self.deletePos(playerPosX, playerPosY)
		
		#add new player
		self.addPlayer(x, y)
		
		self.update(0.1,True)
		print("* ============ WIN ============ *")
		
		self.win = True
		self.saveQTable()
		
		return True
		
	def gameOver(self, nextPosX: int, nextPosY: int) -> bool:
		"""
		游戏失败判定
		"""
		tempPos = posStr(nextPosX, nextPosY)
		if self.map[tempPos] == "X":
			return True
			
		return False
	
	def gameOverUI(self, x, y):
		"""
		踩到陷阱结束界面
		"""
		#delete orignal player
		playerPosX, playerPosY = self.getPlayerPos()
		self.deletePos(playerPosX, playerPosY)
		
		#add new player
		self.addPlayer(x, y)
		
		self.update(0.1,True)
		print("* ============ GAME OVER ============ *")
		
		self.over = True
		self.saveQTable()
		
		return True
	
	def update(self, interval: float, end = False):
		"""
		定时Call
		"""
		#clear screen
		os.system("cls")
		
		#draw information
		#	map player trap end
		self.mapPrint()
		
		#get next move
		nextDir = ""
		if end == False:
			
			# update player to new position
			nextDir = self.playerMove()
		
		#print other information
		print("")
		print("="*20)
		#print("Run Time = %.2fs"%(time()-start))
		print("Next direction = %s"%(nextDir))
		
		#sleep
		sleep(interval)
		
	
	def deletePos(self,x: int, y: int):
		"""
		根据坐标删除map中的元素
		"""
		
		assert x < self.width and x >= 0 ,"addEnd is not in width [0,%d]"%self.width
		assert y < self.height and y >= 0,"addEnd is not in height [0,%d]"%self.height
		
		tempP = posStr(x,y)
		self.map[tempP] = " "
		
	def randomMove(self, curPosX: int,curPosY: int):
		"""
		随机选择方向
		return 
		"""
		#get next move action
		nextDir = random.choice(list(self.directions.keys())) 
		diffX,diffY= self.directions[nextDir]
		
		#loop when is not legal
		while self.legalMove(curPosX + diffX, curPosY + diffY) == False:
			nextDir = random.choice(list(self.directions.keys())) 
			diffX,diffY= self.directions[nextDir]
		
		return nextDir, curPosX + diffX, curPosY + diffY
		
	
	def QMove(self, curPosX: int,curPosY: int):
		"""
		根据Q table采取的移动方式
		"""
		
		#------------------------
		#step 1 get all legal move position
		
		#define a empty list to store all legal move
		legalActions = []
		legalDirs = []
		for k in self.directions:
			tempDx, tempDy = self.directions[k]
			nextPosX = curPosX + tempDx
			nextPosY = curPosY + tempDy
			if self.legalMove(nextPosX,nextPosY):
				
				tempAction = [nextPosX, nextPosY]
				
				legalActions.append(tempAction.copy())
				
				legalDirs.append(k)
		
		#------------------------
		#step 2 take action that make next state and current state max Qvalue
		max_ = -9999
		nextPlayerX = -1
		nextPlayerY = -1
		nextDir = ""
		index = 0
		nextState = -1
		for nextX, nextY in legalActions:
			act = legalDirs[index]
			tempKey1 = QPosStr(curPosX, curPosY, act)
			nextQValue = self.qtable.Qtable[tempKey1]
			
			if nextQValue > max_:
				max_ = nextQValue
				nextPlayerX = nextX
				nextPlayerY = nextY
				nextDir = legalDirs[index]
				nextState = tempKey1
				#print("---->dir = %s, value = %.3f"%(nextDir,diffQvalue))
			index += 1
			
			#if self.currentState == "":
			#	self.currentState = nextState
			
			#update Q
			#self.updateQTable(nextPlayerX, nextPlayerY)
		self.currentState = nextState
		return nextDir, nextPlayerX, nextPlayerY, nextState
		
	def updateQTable(self, curPosX: int, curPosY: int):
		"""
		更新Q表
		"""
		
		#REWARD?
		self.updateReward(curPosX, curPosY)
		
		#------------------------
		#get current Q
		tempCurState = self.currentState
		currentQValue = self.qtable.Qtable[tempCurState]
		
		#------------------------
		#get next Q
		nextDir, nextPlayerX, nextPlayerY, nextState = self.QMove(curPosX, curPosY)
		tempNextState = nextState
		nextQValue = self.qtable.Qtable[tempNextState]
		
		reward = self.currentReward
		gamma = self.qtable.gamma
		alpha = self.qtable.alpha
		
		tempV = (nextQValue * gamma - currentQValue + reward) * alpha + currentQValue
		tempV = round(tempV,5)
		
		self.qtable.Qtable[tempCurState] = tempV
		
		
	
	def updateReward(self, nextPosX: int, nextPosY: int):
		"""
		更新当前奖励
		"""
		tempPos = posStr(nextPosX, nextPosY)
		type = self.map[tempPos]
		self.currentReward = self.rewards[type]
	
	
	def getQValue(self,x,y,action):
		"""
		return Q value
		"""
		q = QPosStr(x,y,action)
		return self.qtable.Qtable[q]
	
	
	def getQRange(self):
		"""
		return Q in max and min
		"""
		q = list(self.qtable.Qtable.values())
		
		return max(q), min(q)
	
	def playerMove(self):
		"""
		玩家移动
		"""
		#get current player Pos
		playerPosX, playerPosY = self.getPlayerPos()
		
		#------------------------------------
		#get next direction, you can replace what your want code
		#nextDir,nextPlayerX,nextPlayerY = self.randomMove(playerPosX, playerPosY)
		nextDir,nextPlayerX,nextPlayerY,nextState = self.QMove(playerPosX, playerPosY)
		
		self.updateQTable(nextPlayerX, nextPlayerY)
		
		#GAME WIN?
		if self.gameWin(nextPlayerX,nextPlayerY):
			self.gameWinUI(nextPlayerX, nextPlayerY)
			return ""
		
		#GAME OVER?
		elif self.gameOver(nextPlayerX,nextPlayerY):
			self.gameOverUI(nextPlayerX, nextPlayerY)
			return ""
		
		
		#delete orignal player
		self.deletePos(playerPosX, playerPosY)
		
		#add new player
		self.addPlayer(nextPlayerX, nextPlayerY)
		
		return nextDir


if __name__ == "__main__":
	
	#-----------------
	# define row and col
	HEIGHT = 6
	WIDTH = 7
	
	#-----------------
	# set rounds
	for j in range(5):
		m=Map(WIDTH,HEIGHT)
		
		# set steps
		for i in range(50):
			m.update(0.1)
			if m.win or m.over:
				m=Map(WIDTH,HEIGHT)
		
		# save Q table
		m.saveQTable()
	
	
	
	
	
