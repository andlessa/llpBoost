#!/usr/bin/env python3

import os,glob
import numpy as np
import pandas as pd
import pyslha
import tempfile,gzip,pylhe
import fastjet


def getJets(events,nevents,dr=0.4,etamax=4.5,
            pdgs=[1,-1,2,-2,3,-3,4,-4,21]):

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 
                                   dr)

    for event in events:
        quarks = [ptc for ptc in events.particles 
                  if int(ptc.id) in pdgs]
        jetArray = [fastjet.PseudoJet(q.Px,q.Py,q.Pz,q.E) for q in quarks if abs(q.Eta) < etamax]
        for ij,j in enumerate(jetArray):
            j.set_user_index(quarks[ij].PID)
        cluster = fastjet.ClusterSequence(jetArray, jetdef)


class ParticleList(object):
    """
    Convenience class for holding a list of Particle objects
    with the same PDG extracted from LHE events.
    """

    def __init__(self,pdg=None) -> None:
        self.particleList = []
        self.ieventList = []
        self.pdg = pdg

    def add(self,ptc,ievent):
        if ptc.PID != self.pdg:
            raise TypeError("Trying to add particle with wrong pdg.")
        self.particleList.append(ptc)
        self.ieventList.append(ievent)

    def __getattr__(self, attr):

        # If calling another special method, return default (required for pickling)
        if (attr.startswith('__') and attr.endswith('__')) or attr in dir(self):
            return self.__getattribute__(attr)

        try:
            val = [getattr(particle, attr) for particle in self.particleList]
            return val
        except AttributeError:
            raise AttributeError("Attribute %s not found in particles" % attr)



class ParticleDict(object):

    def __init__(self,pdgs,labels=None) -> None:
        if not labels:
            labels = pdgs[:]
        self.particleDict = {label : ParticleList(pdg) 
                             for label,pdg in zip(labels,pdgs)}

    def __getitem__(self, label : int) -> ParticleList:
        return self.particleDict[label]

    @classmethod
    def fromEvents(cls,events,nevents,pdgs,status=[1],labels=None):
        pDict = ParticleDict(pdgs)
        for ievent,event in enumerate(events):
            weightPB = event.eventinfo.weight/nevents
            eventDict = {pdg : [] for pdg in pdgs}
            for ptc in event.particles:
                if ptc.id not in pdgs:
                    continue
                if ptc.status not in status:
                    continue
                ptc.PID = int(ptc.id)
                p = np.sqrt(ptc.px**2 + ptc.py**2 + ptc.pz**2)
                ptc.PT = np.sqrt(ptc.px**2 + ptc.py**2)
                if not ptc.PT: # Only for incoming partons
                    ptc.Eta = None
                    ptc.Phi = None
                else:
                    ptc.Eta = (1./2.)*np.log((p+ptc.pz)/(p-ptc.pz))        
                    ptc.Phi = np.arctan2(ptc.py,ptc.px)
                ptc.Px = ptc.px
                ptc.Py = ptc.py
                ptc.Pz = ptc.pz
                ptc.E = ptc.e
                ptc.Beta = p/ptc.E
                eventDict[ptc.PID].append(ptc)
            # Assign a weight for each particle, so
            # the total weight matches
            for pdg,pList in eventDict.items():
                for p in pList:
                    p.weight = weightPB/len(pList)
                    pDict[pdg].add(p,ievent)

        # Convert PDG to labels
        if labels:
            for pdg,label in zip(pdgs,labels):
                pDict.particleDict[label] = pDict.particleDict.pop(pdg)

        return pDict
    

def getLHEevents(fpath):
    """
    Reads a set of LHE files and returns a dictionary with the file labels as keys
    and the PyLHE Events object as values.
    """

    # It is necessary to remove the < signs from the LHE files (in the generate line) before parsing with pylhe
    fixedFile = tempfile.mkstemp(suffix='.lhe')
    os.close(fixedFile[0])
    fixedFile = fixedFile[1]
    with  gzip.open(fpath,'rt') as f:
        data = f.readlines()
        with open(fixedFile,'w') as newF:
            for l in data:
                if 'generate' in l:
                    continue
                newF.write(l)
    events = list(pylhe.read_lhe_with_attributes(fixedFile))
    nevents = pylhe.read_num_events(fixedFile)
    os.remove(fixedFile)
    return nevents,events

def getInfo(f):

    procDict = {
                'p p > x1 n1' : r'$p p \to \tilde{\chi}_1^\pm \tilde{\chi}_1^0$',
                'p p > go go' : r'$p p \to \tilde{g} \tilde{g}$',
                'p p > x1 x1' : r'$p p \to \tilde{\chi}_1^\pm \tilde{\chi}_1^\mp$'
                }
    
    banner = list(glob.glob(os.path.join(os.path.dirname(f),'*banner*')))[0]
    with open(banner,'r') as bannerF:
        bannerData = bannerF.read()
    
    # Get process data:
    processData = bannerData.split('<MGProcCard>')[1].split('</MGProcCard>')[0]

    # Get model
    model = processData.split('Begin MODEL')[1].split('End   MODEL')[0]
    model = model.split('\n')[1].strip()

    # Get process
    proc = processData.split('Begin PROCESS')[1].split('End PROCESS')[0]
    proc = proc.split('\n')[1].split('#')[0].strip()
    if proc in procDict:
        proc = procDict[proc]
    
    # Get parameters data:
    parsData = bannerData.split('<slha>')[1].split('</slha>')[0]
    parsSLHA = pyslha.readSLHA(parsData)
    
    if 1000024 in parsSLHA.blocks['MASS']:
        mG = parsSLHA.blocks['MASS'][1000021]
        mC1 = parsSLHA.blocks['MASS'][1000024]
        mN1 = parsSLHA.blocks['MASS'][1000022]
    else:
        mC1 = 0.0
        mN1 = 0.0
        mG = 0.0
    

    
    # Get event data:
    eventData = bannerData.split('<MGGenerationInfo>')[1].split('</MGGenerationInfo>')[0]
    nEvents = eval(eventData.split('\n')[1].split(':')[1].strip())
    xsec = eval(eventData.split('\n')[2].split(':')[1].strip())

    fileInfo = {'model' : model, 'process' : proc, 
                'mC1' : mC1, 'mN1' : mN1,'mGluino' : mG,
               'xsec (pb)' : xsec, 'MC Events' : nEvents, 'file' : f}
    
    return fileInfo

