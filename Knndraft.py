import numpy as np
import time
#for now I random create training data, testdata and traininglabel
#I expect traininglabel as an array length 60,000
#I expect trainingdata as an matrix 1,680,000 (60000*28)*28
#I expect testdata as an matrix 280,000(10000*28)*28 
trainingdata=np.random.randint(0,2,(1680000,28))
testdata=np.random.randint(0,2,(280000,28))
########################subsampling feature from 28*28 to 14*14#######################
sstrainingdata=trainingdata.reshape(-1,1680000,28)[:,::2,::2].reshape(60000,14*14)
sstestdata=testdata.reshape(-1,280000,28)[:,::2,::2].reshape(10000,14*14)
######################################################################################
trainingdata=trainingdata.reshape(60000,28*28)
testdata=testdata.reshape(10000,28*28)
traininglabel=np.random.randint(0,10,60000)
testres=[]
for i in range(len(sstestdata)):
	time1=time.time()
	thisp=np.tile(sstestdata[i],(len(sstrainingdata),1))
	thisp=((thisp-sstrainingdata)**2)**0.5
	thisp=thisp.T
	thisp=sum(thisp)
	time2=time.time()
	rank=np.argsort(thisp)
	firstten=[]
	for j in range(10):#k
		firstten.append(traininglabel[rank[j]])
	testres.append(np.bincount(np.array(firstten)).argmax())
	time3=time.time()
tstime=tstime+(time3-time2)
tctime=tctime+(time2-time1)
#for each predict, sort spand 0.004 sec
#for each predict, calculation spend 0.55 sec
print("sort time:", tctime)
print("calculate time", tstime)
#I expect predict 10,000 pictures less than 2 hours, which is acceptable.
