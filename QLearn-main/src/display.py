import pygame as pg
import random
from QLearn import *
from time import sleep

def drawColorBarRedBlue(area,maxValue=100, minValue=0 , value=50, width = 20, height=400):
	"""
	Draw a color bar
	area: pg.Surface
	
	return a alpha of map to alpha space
	"""
	global ALPHAs
	#tail rect
	tailWidth = width
	tailHeight = 10
	bins = int(height / tailHeight)
	intervalAlpha = 12
	
	# alpha list
	alpha1 = [255-intervalAlpha*i for i in range(bins//2)]
	alpha2 = alpha1.copy()
	alpha2.reverse()
	alphas = alpha1 + alpha2
	
	# alpha block
	maxAlpha = 255
	minAlpha = alpha1[-1]
	rangeAlpha = (maxAlpha - minAlpha)*2
	numAlpha = len(alphas)
	
	# color block
	tailColors = [RED if i<bins//2 else BLUE for i in range(numAlpha)]
	
	# value block
	valueBlock = (maxValue - minValue) / numAlpha
	valueIndexPos = int(value // valueBlock)
	
	area.fill(WHITE)
	for i in range(1,bins):
		tailSurface = pg.Surface((tailWidth,tailHeight))
		
		color = tailColors[i-1]
		alpha = alphas[i-1]
		
		# map to sub-
		tailSurface.fill(color)
		tailSurface.set_alpha(alpha)
		tailArea = [10,10+(i-1)*tailHeight, tailWidth, tailHeight]
		area.blit(tailSurface,tailArea)
	
	
	fontSurface = FONT.render("%d局"%ROUNDS,False,BLACK,WHITE)
	area.blit(fontSurface,[xMargin ,400,BLOCK_SIZE,BLOCK_SIZE])
	fontSurface = FONT.render("%d步"%STEPS,False,BLACK,WHITE)
	area.blit(fontSurface,[xMargin ,400+BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE])
	
	return tailColors[valueIndexPos], alphas[valueIndexPos]


def drawUDArrow(mainArea, subArea, x, y):
	"""
	绘制上下指向的箭头
	
	-------------------------------
	mainArea: 窗口的主绘图平面
	subArea:  窗口的副绘图平面
	x: 对应 GRID_MAT 的列索引
	y: 对应 GRID_MAT 的行索引
	"""
	
	# get coordinate of Qtable
	xQUp = x // 2
	yQUp = (y+1) // 2
	
	
	# set size of arrow
	h = (BLOCK_SIZE-2*yMargin) / 6
	w = (BLOCK_SIZE-2*xMargin) / 3
	
	
	#-----------------------------------
	#up Arrow
	# create a surface
	UparrowSurface = pg.Surface((BLOCK_SIZE/2,BLOCK_SIZE))
	
	# get q value
	seed = m.getQValue(xQUp, yQUp, "up")
	
	# get range of total Q value
	max_, min_ = m.getQRange()
	
	# get color and alpha when map to a colorbar
	c,a = drawColorBarRedBlue(subArea,maxValue=max_,minValue=min_,value = seed)
	
	# fill arrow bg color
	UparrowSurface.fill(WHITE)
	
	# set alpha
	UparrowSurface.set_alpha(a)
	
	
	# draw a up arrow
	startX = xMargin + h
	startY = 3*w + yMargin
	pg.draw.polygon(UparrowSurface,c,[
		(startX,startY),
		(startX,startY-2*w),
		(startX-h,startY-2*w),
		(startX+0.5*h,startY-3*w),
		(startX+2*h,startY-2*w),
		(startX+1*h,startY-2*w),
		(startX+h,startY),
			],0)
	
	# place it on main screen
	X = x * BLOCK_SIZE
	Y = y * BLOCK_SIZE
	mainArea.blit(UparrowSurface,[X,Y,3*w,3*h])
	
	
	#--------------------------------
	#down Arrow
	# get coordinate of Qtable
	xQDown = xQUp
	yQDown = yQUp - 1
	
	# create a surface
	DownarrowSurface = pg.Surface((BLOCK_SIZE/2,BLOCK_SIZE))
	
	# get Q value
	seed = m.getQValue(xQDown, yQDown, "down")
	
	# get Q range
	max_, min_ = m.getQRange()
	
	# get color and alpha when map to colorbar 
	c,a = drawColorBarRedBlue(subArea,maxValue=max_,minValue=min_,value = seed)
	
	# fill bg color
	DownarrowSurface.fill(WHITE)
	
	# set alpha
	DownarrowSurface.set_alpha(a)
	
	
	# draw down
	startX = xMargin + h
	startY = yMargin
	pg.draw.polygon(DownarrowSurface,c,[
		(startX,startY),
		(startX,startY+2*w),
		(startX-h,startY+2*w),
		(startX+0.5*h,startY+3*w),
		(startX+2*h,startY+2*w),
		(startX+h,startY+2*w),
		(startX+h,startY),
			],0)
	
	# place it on screen
	mainArea.blit(DownarrowSurface,[X+BLOCK_SIZE*0.5,Y,3*w,3*h])
	
	
def drawLRArrow(mainArea, subArea, x, y):
	"""
	绘制左右指向的箭头
	
	-------------------------------
	mainArea: 窗口的主绘图平面
	subArea:  窗口的副绘图平面
	x: 对应 GRID_MAT 的列索引
	y: 对应 GRID_MAT 的行索引
	"""
	
	# get coordinate of Qtable
	xQRight = (x-1) // 2
	yQRight = y//2
	
	
	# set size of arrow
	h = (BLOCK_SIZE-2*yMargin) / 6
	w = (BLOCK_SIZE-2*xMargin) / 3
	
	
	#-------------------------
	#Right Arrow
	
	# create a surface
	RarrowSurface = pg.Surface((BLOCK_SIZE,3*h + 2*yMargin))
	
	# get Q value
	seed = m.getQValue(xQRight, yQRight, "right")
	
	# get range of Q
	max_, min_ = m.getQRange()
	
	# get color and alpha when map to colorbar
	c,a = drawColorBarRedBlue(subArea,maxValue=max_,minValue=min_,value = seed)
	
	# fill bg color of arrow
	RarrowSurface.fill(WHITE)
	
	# set alpha of arrow
	RarrowSurface.set_alpha(a)
	
	
	# draw arrow
	startX = xMargin
	startY = h + yMargin
	pg.draw.polygon(RarrowSurface,c,[
		(startX,startY),
		(startX+2*w,startY),
		(startX+2*w,startY-h),
		(startX+3*w,startY+0.5*h),
		(startX+2*w,startY+2*h),
		(startX+2*w,startY+1*h),
		(startX,startY+1*h),
			],0)
	
	
	# place arrow on screen
	X = x * BLOCK_SIZE
	Y = y * BLOCK_SIZE
	mainArea.blit(RarrowSurface,[X, Y, 3*w, 3*h])
	
	
	#---------------------------------
	#Left Arrow
	xQLeft = xQRight + 1
	yQLeft = yQRight
	
	# create a surface
	LarrowSurface = pg.Surface((BLOCK_SIZE,3*h + 2*yMargin))
	
	# get Q value
	seed = m.getQValue(xQLeft, yQLeft, "left")
	
	# get Q range
	max_, min_ = m.getQRange()
	
	# get color and alpha when map to colorbar
	c,a = drawColorBarRedBlue(subArea,maxValue=max_,minValue=min_,value = seed)
	
	# fill bg color of arrow
	LarrowSurface.fill(WHITE)
	
	# set alpha
	LarrowSurface.set_alpha(a)
	
	
	# draw arrow
	startX = 3*w + xMargin
	startY = h + yMargin
	pg.draw.polygon(LarrowSurface,c,[
		(startX,startY),
		(startX-2*w,startY),
		(startX-2*w,startY-h),
		(startX-3*w,startY+0.5*h),
		(startX-2*w,startY+2*h),
		(startX-2*w,startY+1*h),
		(startX,startY+1*h),
			],0)
	
	# place it on screen
	mainArea.blit(LarrowSurface,[X,Y+BLOCK_SIZE*0.5,3*w,3*h])
	
def drawGameEle(mainSurface,x,y):
	"""
	绘制 player 世界中的标识：
		1. player："P"
		2. Trap：  "X"
		1. End：   "E"
	-------------------------------
	mainSurface: 窗口的主绘图平面
	x: 对应 GRID_MAT 的列索引
	y: 对应 GRID_MAT 的行索引
	"""
	
	# translate index into Q format 
	# because index of GRID_MAT isn't equal Q table's, GRID_MAT is (2*n - 1) in row and col, n is number of rows or cols in Q table 
	# So, index of GRID_MAT value rounded by dividing by 2 equal Q table
	x2,y2 = x//2, y//2
	
	# Trap?
	if m.isTrap(x2,y2):
		tempChar = "X"
		fg = GREEN
	
	# Player?
	elif m.isPlayer(x2,y2):
		tempChar = "P"
		fg = BLACK
	
	# End?
	elif m.isEnd(x2,y2):
		tempChar = "E"
		fg = BLACK
	
	# others
	else:
		return
	
	# identification of element
	fontSurface = FONT.render(tempChar,False,fg,WHITE)
	
	# place on mainSurface
	mainSurface.blit(fontSurface,[xMargin + x*BLOCK_SIZE,yMargin+ y*BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE])

def drawMap(mainSurface,subSurface):
	"""
	根据 GRID_MAT 数组中不同的数字绘制不同的图形元素
	其中：
		1 表示player可以移动到的空间
		2 表示左右指向的箭头
		3 表示上下指向的箭头
		0 表示空
	"""
	for i in range(GRID_TOTAL_COL):
		for j in range(GRID_TOTAL_ROW):
			
			# draw position of player can move to
			if GRID_MAT[j][i] == 1:
				
				# draw a boundary
				pg.draw.rect(mainSurface,BLACK,[i*BLOCK_SIZE,j*BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE],1)
				
				# draw element of game, like player trap and end
				drawGameEle(mainSurface,i,j)
				
			# draw a direction arrow, Left to Right and Right to Right
			elif GRID_MAT[j][i] == 2:
				drawLRArrow(mainSurface,subSurface,i,j)
			
			# draw a direction arrow, Up to Down and Down to Up
			elif GRID_MAT[j][i] == 3:
				drawUDArrow(mainSurface,subSurface,i,j)
def Draw():
	"""
	绘制主函数
	内容包括：
		1. 将这个窗口界面分割成两个部分，main and sub
		2.其中main区域用于绘制player的世界，如可以移动的区域，陷阱，终点，方向箭头等
	"""
	
	#------------------
	# Split screen to two area
	# main area and sub-area
	
	# main position
	mainWidth = WINDOW_HEIGHT - 2*xMargin
	mainHeight = WINDOW_HEIGHT - 2*yMargin
	mainArea = [0 + xMargin, 0 + yMargin, mainWidth, mainHeight]
	mainSurface = pg.Surface((mainWidth, mainHeight))
	
	# sub-area position
	subWidth = SUB_SURFACE_WIDTH - 2*xMargin
	subHeight = WINDOW_HEIGHT - 2*yMargin
	subArea = [mainWidth + 3*xMargin, 0 + yMargin, mainWidth + 3*xMargin + subWidth, subHeight]
	subSurface = pg.Surface((subWidth, subHeight))
	
	
	# main and sub background color
	mainBgColor = WHITE
	subBgColor = WHITE
	mainSurface.fill(mainBgColor)
	subSurface.fill(subBgColor)
	
	#----------------------
	# start draw world of player
	drawMap(mainSurface,subSurface)
	
	#-------------------
	# place them to screen
	screen.blit(mainSurface,mainArea)
	screen.blit(subSurface,subArea)
	pg.draw.line(screen,BLACK,(mainWidth+2*xMargin,0+yMargin),(mainWidth+2*xMargin,mainHeight+yMargin))
	
	pg.display.flip()

#-----------------------
#common pygame configure

# pggame initialization
pg.init()

# subSurface width  height equal to window's
SUB_SURFACE_WIDTH = 100

# margin size of width and height
xMargin = 5
yMargin = 5

# window width and height
WINDOW_WIDTH = 700
WINDOW_HEIGHT = WINDOW_WIDTH - SUB_SURFACE_WIDTH

# define a size of grid 
GAME_GRID_ROW = 7 #player ground
GAME_GRID_COL = 7 #player ground


# define a size of block
MAX_ = GAME_GRID_ROW if GAME_GRID_ROW > GAME_GRID_COL else GAME_GRID_COL
BLOCK_SIZE = (WINDOW_HEIGHT-2*xMargin) / (2*MAX_ - 1)


#--------------------------
# define a GRID_MAT
GRID_TOTAL_ROW = 2*GAME_GRID_ROW - 1
GRID_TOTAL_COL = 2*GAME_GRID_COL - 1
GRID_MAT = [[0 for j in range(GRID_TOTAL_COL)] for i in range(GRID_TOTAL_ROW)]

#Fill mat
for i in range(GRID_TOTAL_ROW):
	for j in range(GRID_TOTAL_COL):
		
		# available position
		if i%2 == 0 and j %2 == 0:
			GRID_MAT[i][j] = 1
		
		# left & right arrow
		elif i%2 == 0 and j %2 != 0:
			GRID_MAT[i][j] = 2
		
		# up & down arrow
		elif i%2 != 0 and j %2 == 0:
			GRID_MAT[i][j] = 3
		
		# blank
		elif i%2 != 0 and j %2 != 0:
			GRID_MAT[i][j] = 0


#-----------------------
# define a window
screen=pg.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))

# fill bg of window
screen.fill((255,255,255))

#-----------------------
# define a clock
clock = pg.time.Clock()


#-----------------------
# define color
RED   = (255,0,0)
BLUE  = (0,0,255)
GREEN = (0,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)

#-----------------------
# dedine a font
FONT=pg.font.SysFont("华文楷体",28,bold=True,italic=True)


if __name__ == "__main__":
	
	#----------------
	#let's show start :p
	
	# define number of step in round
	STEPS = 0
	
	# define number of round
	ROUNDS = 1
	
	# create a map
	m=Map(GAME_GRID_COL,GAME_GRID_ROW)
	
	# loop
	while True:
		
		# destroyed window
		for event in pg.event.get():
			if event.type==12:
				pg.quit()
				exit()
		
		# start update map
		m.update(0.05)
		STEPS += 1
		if m.win or m.over:
			Draw()
			m=Map(GAME_GRID_COL,GAME_GRID_ROW)
			m.saveQTable()
			ROUNDS += 1
			STEPS = 0
			sleep(1)
			
		# start draw it
		Draw()
		
