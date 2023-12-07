#!/usr/bin/env python3

import os,glob
import numpy as np
import pandas as pd
import pyslha
import time
import progressbar as P
import tempfile,gzip,pylhe
import fastjet

class ParticleList(object):

    def __init__(self) -> None:
        self.particleList = []
        self.weightList = []

    @classmethod
    def fromEvents(cls,events,nevents,pdgs,status=[1]):
        pList = ParticleList()
        for event in events:
            weightPB = event.eventinfo.weight/nevents
            particlesInEvent = []
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
                particlesInEvent.append(ptc)
            pList.particleList += particlesInEvent[:]
            pList.weightList += [weightPB/len(particlesInEvent)]*len(particlesInEvent)

        return pList
    
    def __getattr__(self, attr):

        # If calling another special method, return default (required for pickling)
        if (attr.startswith('__') and attr.endswith('__')) or attr in dir(self):
            return self.__getattribute__(attr)

        try:
            val = [getattr(particle, attr) for particle in self.particleList]
            return val
        except AttributeError:
            raise AttributeError("Attribute %s not found in particles" % attr)


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

def getWinos(event,etamax=2.0,pTmin=355.0):

    # Add information to particles:
    for ptc in event.particles:
        ptc.daughters = []
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
        
    for ptc in event.particles:
        for mom in ptc.mothers():
            mom.daughters.append(ptc)
    
    # Get charginos and neutralinos
    charginos = {}
    neutralinos = {}
    for ptc in event.particles:        
        if abs(ptc.PID) == 1000024:
            charginos[ptc.PID] = ptc
        elif abs(ptc.PID) == 1000022:
            neutralinos[ptc.PID] = ptc # Store only the last top/anti-top            
    
    
    return charginos,neutralinos

def getBoost(nevents,events):
    """
    Reads a PyLHE Event object and extracts the ttbar invariant
    mass for each event.
    """

    boostC1 = []
    boostN1 = []
    weights = []
    for ev in events:        
        weightPB = ev.eventinfo.weight/nevents
        weightAndError = np.array([weightPB,weightPB**2])
        charginos,neutralinos = getWinos(ev)
        for c1 in charginos.values():
            boostC1.append(c1.Beta)
        for n1 in neutralinos.values():
            boostN1.append(n1.Beta)

        weights.append(weightAndError)
    
    weights = np.array(weights)
    boostN1 = np.array(boostN1)
    boostC1 = np.array(boostC1)    
    
    return boostC1,boostN1,weights


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


def getRecastData(inputFiles,weightMultiplier=1.0,skipParameters=[]):

    
    # Filter files (if needed)
    if not skipParameters:
        selectedFiles = inputFiles[:]
    else:
        selectedFiles = []
        for f in inputFiles:
            # print('\nReading file: %s' %f)
            fileInfo = getInfo(f)
            parInfo = (fileInfo['mST'],fileInfo['mChi'],fileInfo['yDM'], 
                        fileInfo['mT'], fileInfo['model'], fileInfo['process'])
            if parInfo in skipParameters:
                continue
            selectedFiles.append(f)
        print('Skipping %i files' %(len(inputFiles)-len(selectedFiles)))

    if not selectedFiles:
        return None
    
    allData = []

    progressbar = P.ProgressBar(widgets=["Reading %i Files: " %len(selectedFiles), 
                            P.Percentage(),P.Bar(marker=P.RotatingMarker()), P.ETA()])
    progressbar.maxval = len(selectedFiles)
    progressbar.start()
    nfiles = 0

    for f in selectedFiles:
        # print('\nReading file: %s' %f)
        fileInfo = getInfo(f)
        # Get events:
        nevents,events = getLHEevents(f)
        data = getPTThist(nevents,events,weightMultiplier=weightMultiplier)
        # Create a dictionary with the bin counts and their errors
        dataDict = fileInfo
        bins_left = data[:,0]
        bins_right = data[:,1]
        w = data[:,2]
        wError = data[:,3]    
        for ibin,b in enumerate(bins_left):
            label = 'bin_%1.0f_%1.0f'%(b,bins_right[ibin])
            dataDict[label] = w[ibin]
            dataDict[label+'_Error'] = wError[ibin]
        allData.append(dataDict)
        nfiles += 1
        progressbar.update(nfiles)

    progressbar.finish()
    return allData


if __name__ == "__main__":
    
    import argparse    
    ap = argparse.ArgumentParser( description=
            "Run the recasting for ATLAS-TOPQ-2019-23 using one or multiple MadGraph (parton level) LHE files. "
            + "If multiple files are given as argument, add them."
            + " Store the SR bins in a pickle (Pandas DataFrame) file." )
    ap.add_argument('-f', '--inputFile', required=True,nargs='+',
            help='path to the LHE event file(s) generated by MadGraph.', default =[])
    ap.add_argument('-o', '--outputFile', required=False,
            help='path to output file storing the DataFrame with the recasting data.'
                 + 'If not defined, will use the name of the first input file', 
            default = None)
    ap.add_argument('-w', '--weightMultiplier', required=True, type=float,
            help='Factor used to multiply the weights (in case events were generated with specific top decays in each branch)', default =[])
    ap.add_argument('-O', '--overwrite', required=False, action='store_true',
                    help='If set, will overwrite the existing output file. Otherwise, it will simply add the points not yet present in the file', default = False)


    t0 = time.time()

    # # Set output file
    args = ap.parse_args()
    inputFiles = args.inputFile
    outputFile = args.outputFile
    weightMultiplier = args.weightMultiplier

    if outputFile is None:
        outputFile = inputFiles[0].replace('.lhe.gz','_atlas_topq_2019_23.pcl')

    if os.path.splitext(outputFile)[1] != '.pcl':
        outputFile = os.path.splitext(outputFile)[0] + '.pcl'


    skipParameters = []
    if os.path.isfile(outputFile):
        if args.overwrite:
            print('Output file %s will be overwritten!' %outputFile)
        else:
            df_orig = pd.read_pickle(outputFile)
            skipParameters = []
            for irow,row in df_orig.iterrows():
                skipParameters.append((row['mST'],row['mChi'],row['yDM'], row['mT'], row['model'], row['process']))
    
    print('-----------------\n Running with weight multiplier = %1.1f\n -------------------------' %weightMultiplier)

    dataDict = getRecastData(inputFiles,weightMultiplier)
    if dataDict:
        # #### Create pandas DataFrame
        df = pd.DataFrame.from_dict(dataDict)
        if os.path.isfile(outputFile) and skipParameters:
            df = pd.concat([df_orig,df])

        # ### Save DataFrame to pickle file
        print('Saving to',outputFile)
        df.to_pickle(outputFile)

    print("\n\nDone in %3.2f min" %((time.time()-t0)/60.))
