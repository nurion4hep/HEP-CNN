#!/usr/bin/env pythnon
import h5py
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from glob import glob
import pandas as pd
import numpy as np

class HEPCNNDataset(Dataset):
    def __init__(self, **kwargs):
        super(HEPCNNDataset, self).__init__()
        self.isLoaded = False
        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "weight", "label", "fileIdx", "suffix"])

    def __getitem__(self, idx):
        if not self.isLoaded: self.load()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = idx-int(offset)

        image  = self.imagesList[fileIdx][idx]
        label  = self.labelsList[fileIdx][idx]
        weight = self.weightsList[fileIdx][idx]
        rescale = self.rescaleList[fileIdx][idx]
        procIdx = self.procList[fileIdx][idx]
        evtWeight = self.evtWeightList[fileIdx][idx]

        return (image, label, weight, rescale, procIdx, evtWeight)

    def __len__(self):
        return int(self.maxEventsList[-1])

    def addSample(self, procName, fNamePattern, weight=1., logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fileNamePattern))
        ##weightValue = weight ## Rename it just to be clear in the codes

        for fName in glob(fNamePattern):
            if not fName.endswith(".h5"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'weight':weight, 'nEvents':0,
                'label':0, ## default label, to be filled later
                'fileName':fName, 'fileIdx':fileIdx,
                'suffix':"", 'sumEvtWeight' : 0
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)

    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label

    def initialize(self, logger=None):
        if logger: logger.update(annotation='Reweights by category imbalance')
        procNames = list(self.sampleInfo['procName'].unique())

        self.labelsList = []
        self.weightsList = []
        self.rescaleList = []
        self.procList = []
        self.imagesList = []
        self.evtWeightList = []
        self.evtWeightSumList = []

        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
            data = h5py.File(fName, 'r', libver='latest', swmr=True)['all_events']
            suffix = "_val" if 'images_val' in data else ""

            images  = data['images'+suffix]
            nEvents = images.shape[0]
            self.sampleInfo.loc[i, 'nEvents'] = nEvents
            self.sampleInfo.loc[i, 'suffix'] = suffix

            weightValue = self.sampleInfo.loc[i, 'weight']
            weights = torch.ones(nEvents, dtype=torch.float32, requires_grad=False)*weightValue
            self.weightsList.append(weights)

            evtWeights = torch.tensor(data['weights'+suffix], dtype=torch.float32, requires_grad=False)
            self.evtWeightList.append(evtWeights)
            self.sampleInfo.loc[i, 'sumEvtWeight'] = torch.sum(evtWeights).item()
            self.evtWeightSumList.append(torch.sum(evtWeights).item())

            ## set label and weight
            label = self.sampleInfo['label'][i]
            labels = torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*label
            self.labelsList.append(labels)
            #weight = self.sampleInfo['weight'][i]
            #weights = torch.ones(nEvents, dtype=torch.float32, requires_grad=False)*weight
            #self.weightsList.append(weights)
            self.rescaleList.append(torch.ones(nEvents, dtype=torch.float32, requires_grad=False))
            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)

        print(self.sampleInfo)

        ## Compute cumulative sums of nEvents, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvents'])))

        ## Compute sum of weights for each label categories
        sumWByLabel = {}
        sumEByLabel = {}
        rescaleFactor = {}
        for label in self.sampleInfo['label']:
            label = int(label)
            w = self.sampleInfo[self.sampleInfo.label==label]['weight']
            e = self.sampleInfo[self.sampleInfo.label==label]['nEvents']
            sumEvtW = self.sampleInfo[self.sampleInfo.label==label]['sumEvtWeight']
            sumWByLabel[label] = (w*sumEvtW).sum()
            sumEByLabel[label] = e.sum()
        ## Find overall rescale for the data imbalancing problem - fit to the category with maximum entries
        maxSumELabel = max(sumEByLabel, key=lambda key: sumEByLabel[key])
        maxWMaxSumELabel = self.sampleInfo[self.sampleInfo.label==maxSumELabel]['weight'].max()
        minWMaxSumELabel = self.sampleInfo[self.sampleInfo.label==maxSumELabel]['weight'].min()
        avgWgtMaxSumELabel = sumWByLabel[maxSumELabel]/sumEByLabel[maxSumELabel]

        ## Find overall rescale for the data imbalancing problem - fit to the category with maximum entries
        #### Find rescale factors - make weight to be 1 for each cat in the training step
        for fileIdx in self.sampleInfo['fileIdx']:
            label = self.sampleInfo.loc[self.sampleInfo.fileIdx==fileIdx, 'label']
            for l in label: ## this loop runs only once, by construction.
                self.rescaleList[fileIdx] *= ( sumEByLabel[maxSumELabel]/sumWByLabel[l] )
                #self.rescaleList[fileIdx] *= ( (1/sumWByLabel[l])*(sumEByLabel[l])*(sumEByLabel[maxSumELabel]/sumEByLabel[l]) )
                rescaleFactor[l] = (float(sumEByLabel[maxSumELabel])/float(sumWByLabel[l]))
                #print("@@@ Scale sample label_%d(sumE=%g,sumW=%g)->label_%d, sf=%f" % (l, sumEByLabel[l], sumWByLabel[l], maxSumELabel, sf))
                break ## this loop runs only once, by construction. this break is just for a confirmation
        
        print('-'*80)
        for label in sumWByLabel.keys():
            if(len(self.sampleInfo[self.sampleInfo.label==label]['weight'].unique()) == 1):
                proc = (self.sampleInfo[self.sampleInfo.label==label]['procName'].unique())[0]
                weightfactor = (self.sampleInfo[self.sampleInfo.label==label]['weight'].unique()[0])
                print("Process= %s, Label= %d, sumE= %d, sumW= %g" % (proc, label, sumEByLabel[label], sumWByLabel[label]))
                #print("Rescale factor for process %s, label %d = %g" %(proc, label, rescaleFactor[label]))
                #print("Weight factor for process %s, label %d = %g" %(proc, label, weightfactor))
                #print("(rescale * weight factor) for process %s, label %d = %g" %(proc, label, (rescaleFactor[label]*weightfactor) ))
            else:
                for proc in self.sampleInfo[self.sampleInfo.label==label]['procName'].unique():
                    weightfactor = (self.sampleInfo[self.sampleInfo.label==label][self.sampleInfo.procName==proc]['weight'].unique()[0])
                    print("Process= %s, Label= %d, sumE= %d, sumW= %g" % (proc, label, sumEByLabel[label], sumWByLabel[label]))
                    #print("Rescale factor for process %s, label %d = %g" %(proc, label, rescaleFactor[label]))
                    #print("Weight factor for process %s, label %d = %g" %(proc, label, weightfactor))
                    #print("(rescale * weight factor) for process %s, label %d = %g" %(proc, label, (rescaleFactor[label]*weightfactor) ))
        print('Label with maxSumE:%d' % maxSumELabel)
        print('      maxWeight=%g minWeight=%g avgWeight=%g' % (maxWMaxSumELabel, minWMaxSumELabel, avgWgtMaxSumELabel))
        print('-'*80)

        ## Check the image format
        fNameTemp = self.sampleInfo['fileName'][0]
        suffixTemp = self.sampleInfo['suffix'][0]
        data = h5py.File(fNameTemp, 'r', libver='latest', swmr=True)['all_events']
        self.shape = data['images'+suffix].shape[1:]
        if self.shape[-1] <= 5: ## actual format was NHWC
            self.format = 'NHWC'
            self.height, self.width, self.channel = self.shape
        else:
            self.format = 'NCHW'
            self.channel, self.height, self.width= self.shape

    def load(self):
        if self.isLoaded: return
        for fName, suffix in zip(list(self.sampleInfo['fileName']), list(self.sampleInfo['suffix'])):
            image = h5py.File(fName, 'r', libver='latest', swmr=True)['all_events/images'+suffix]
            self.imagesList.append(image)
        self.isLoaded = True

