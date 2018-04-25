class BoundingBox():

	# topL = (0,0)	#top left coordinate of box
	# botR = (0,0) 	#bottom right coordinate of box
	# mid = (0,0)		#midpoint of the box
	# val = "" 		#value inside box


	# constructor
	def __init__(self, x1, y1, x2, y2, symbol):
		self.topL = (x1,y1)
		self.botR = (x2,y2)
		self.val = symbol
		# self.mid = (0.5*(self.topL[0]+self.botR[0]),0.5*(self.topL[1]+self.botR[1]))
	# set top left coordinate
	def setTopL(self,x, y):
		self.topL = (x,y)
	# set bottom left coordinate
	def setBotR(self, x, y):
		self.botR = (x,y)
	# set value/input variable
	def setVal(self,x):
		self.val = x
	# print value
	def display(self):
		return self.val
	# get **weighted** midpoint
	def getMid(self):
		return (0.5*(0.33*self.topL[0]+0.66*self.botR[0]),
				0.5*(0.33*self.topL[1]+0.66*self.botR[1]))

	# compare two boxes to determine which comes first
	# return true if x is first, false if y is first
	def compare(self,y):
		if(self.getMid()[0] < y.getMid()[0]):
			return 1
		else:
			return 0

	# combine two bounding boxes into a new bounding box
	# CURRENTLY ONLY WORKS FOR HORIZONTALS
	def combine(self,otherBox):
		if(self.compare(otherBox)):
			return BoundingBox(self.topL[0],self.topL[1],otherBox.botR[0],
								otherBox.botR[1],'' + self.val+otherBox.val)
		else:
			return BoundingBox(otherBox.topL[0],otherBox.topL[1],
								self.botR[0],self.botR[1],
								'' + self.val+otherBox.val)

# test
def start():
	a = BoundingBox(10,28,87,140,"5")
	b = BoundingBox(96,91,175,131,"-")
	c = BoundingBox(183,12,266,136,"2")
	d = BoundingBox(282,78,365,144,"+")
	e = BoundingBox(376,1,450,132,"3")
	f = BoundingBox(468,68,516,132,"*")
	g = BoundingBox(527,1,603,126,"4")
	h = BoundingBox(613,66,680,117,"-")
	i = BoundingBox(688,1,792,114,"10")
	a = a.combine(b)
	a = a.combine(c)
	a = a.combine(d)
	a = a.combine(e)
	a = a.combine(f)
	a = a.combine(g)
	a = a.combine(h)
	a = a.combine(i)
	a.display()
	#print("The value of the expression is: ")
	#print(eval(a.val))

#start()
