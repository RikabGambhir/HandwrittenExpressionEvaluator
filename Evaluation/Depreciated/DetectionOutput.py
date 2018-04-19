from BoundingBox import BoundingBox
import math

class DetectionOutput:
	# BoxList = array of bounding boxes

	# constructor
	# BoxList is the "permanent variable"
	# boxListTemp is the mutable version used for computation
	def __init__(self, BoxList):
		self.BoxList = BoxList
		self.boxListTemp = BoxList
	# get the magnitude of vector/tuple from v to w
	def getMagnitude(self,v,w):
		return (math.sqrt((v[0]-w[0])**2+(v[1]-w[1])**2))

	# get the box closest to the point p
	def getFirst(self,p):
		f = None
		minMag = 0
		index = None
		for box in self.boxListTemp:
			if(f == None or (minMag > self.getMagnitude(p,box.getMid()))):
				f = box
				minMag = self.getMagnitude(p,box.getMid())
				index = self.boxListTemp.index(box)
		self.boxListTemp.remove(self.boxListTemp[index]) #remove the box that was found
		return f

	# recursively merges all the boxes in the list into one result box
	def combineAll(self):
		bigBox = self.getFirst((0,0))
		return self.combineAllRec(bigBox)

	# inner recursive function: finds the box closest to aggregate box and merges
	def combineAllRec(self,bigBox):
		if(len(self.boxListTemp) == 0):
			return bigBox
		else:
			bigBox = bigBox.combine(self.getFirst(bigBox.getMid()))
			return self.combineAllRec(bigBox)

# test
def start():
	a = BoundingBox(90,66,191,201,"8")
	b = BoundingBox(214,102,308,132,"-")
	c = BoundingBox(322,65,423,196,"4")
	d = BoundingBox(83,222,431,271,"/")
	e = BoundingBox(97,312,190,402,"2")
	f = BoundingBox(217,319,293,378,"+")
	g = BoundingBox(318,305,391,447,"3")
	testBoxList = [a,b,c,d,e,f,g]
	testDetection = DetectionOutput(testBoxList)
	result = testDetection.combineAll()
	#result = testDetection.getFirst((0,0))

	result.display()

start()



