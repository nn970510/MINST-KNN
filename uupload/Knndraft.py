import numpy as np
import time
#for now I random create training data, testdata and traininglabel
#I expect traininglabel as an array length 60,000
#I expect trainingdata as an matrix 1,680,000 (60000*28)*28
#I expect testdata as an matrix 280,000(10000*28)*28 
class KnnClassifier:
	def __init__(self, trainingsize, traingingdata, traininglabel, testdata, k):
		self.trainingdata=traingingdata[:int(len(traingingdata)*trainingsize)]
		self.traininglabel=traininglabel[:int(len(traininglabel)*trainingsize)]
		self.testdata=testdata
		self.k=k

	def subsampling(self):
		print (len(self.trainingdata),"traingingdata")		
		self.trainingdata=self.trainingdata.reshape(-1,len(self.trainingdata),28)[:,::2,::2]
		self.testdata=self.testdata.reshape(-1,len(self.testdata),28)[:,::2,::2]
		self.testdata=self.testdata[0]
		self.trainingdata=self.trainingdata[0]
		print ("Done for Subsampling, now there are: ")

	def test(self, sb):
		tstime=0
		tctime=0
		if sb!=0:
			self.subsampling()
		print (len(self.trainingdata),"traingingdata")
		print ("each traingdata has", np.size(self.trainingdata,1), "feature")		
		self.testdata=self.testdata.reshape(int(len(self.testdata)/len(self.testdata[0])),len(self.testdata[0])*len(self.testdata[0]))
		self.trainingdata=self.trainingdata.reshape(int(len(self.trainingdata)/len(self.trainingdata[0])),len(self.trainingdata[0])*len(self.trainingdata[0]))
		testres=[]
		print("testdatalen:",len(self.testdata))
		for i in range(len(self.testdata)):
			time1=time.time()
			thisp=np.tile(self.testdata[i],(len(self.trainingdata),1))
			thisp=(thisp-self.trainingdata)**2
			thisp=thisp.T
			thisp=sum(thisp)**0.5
			time2=time.time()
			rank=np.argsort(thisp)
			firstten=[]
			for j in range(self.k):#k
				firstten.append(self.traininglabel[rank[j]])
			testres.append(np.bincount(np.array(firstten).astype(np.int64)).argmax())
			time3=time.time()
			tstime=tstime+(time3-time2)
			tctime=tctime+(time2-time1)
		#for each predict, sort spand 0.004 sec
		#for each predict, calculation spend 0.55 sec
		print("sort time:", tctime)
		print("calculate time", tstime)
		return testres
  
