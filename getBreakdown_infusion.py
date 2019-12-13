#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import os
from pyteomics import mzml
from scipy.integrate import simps
import sys
import numpy as np
from bisect import bisect_left
from matplotlib.backends.backend_pdf import PdfPages
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

def readRawDataFile(filename, maxMass, resolution, useMS1, ppmWidth = 50,offset=0.65):
   #try:
        delete = False
        if ".mzML" not in filename:
            command = "msconvert " + filename + ' --ignoreUnknownInstrumentError > junk.txt'
            os.system(command)
            filename = filename.split("/")[-1]
            filename = filename.replace(".raw", "")
            filename = filename.replace('"', "")
            filename = filename+".mzML"
            delete = True
        reader = mzml.read(filename.replace('"', ""))
        result = []
        ms1Scans = {}
        for temp in reader:
            if temp['ms level'] == 2:
                    tic = np.log10(float(temp["total ion current"]))
                    id = temp["id"].split()[-1].split("=")[-1]
                    #centerMz = temp["precursorList"]["precursor"][0]["isolationWindow"]['isolation window target m/z']
                    centerMz = temp["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"]
                    try:
                        lowerBound = centerMz - temp["precursorList"]["precursor"][0]["isolationWindow"]['isolation window lower offset']
                        upperBound = centerMz + temp["precursorList"]["precursor"][0]["isolationWindow"]['isolation window upper offset']
                    except:
                        lowerBound = centerMz - offset
                        upperBound = centerMz + offset
                    filter = temp["scanList"]["scan"][0]["filter string"]
                    #acquisitionMode = filter.split("@")[0].split()[1]
                    if 'positive scan' in temp:
                        acquisitionMode = "Positive"
                    else:
                        acquisitionMode = "Negative"
                    settings = filter.split("@")[1].split()[0]
                    fragmentMode = settings[:3]
                    NCE = float(settings[3:])
                    rt = temp["scanList"]["scan"][0]["scan start time"]
                    mzs = list(zip(temp["m/z array"],temp["intensity array"]))
                    tempSpecs = []
                    for res in resolution:
                        spectra ={np.round(x[0],res):0 for x in mzs}
                        for x,y in mzs: spectra[np.round(x,res)] += y
                        tempSpecs.append(spectra)
                    result.append({"id":id,"spectra":tempSpecs,"NCE":NCE,"mode":acquisitionMode,"center m/z":
                                       centerMz,"lower m/z":lowerBound,"higher m/z":upperBound,"rt":rt,"signal":tic})
            elif useMS1 and temp['ms level'] == 1:
                ms1Scans[temp["scanList"]["scan"][0]["scan start time"]] = {mz: i for mz, i in zip(temp["m/z array"], temp["intensity array"])}
            
        reader.close()
        if len(ms1Scans) > 0 and useMS1:
            rts = list(ms1Scans.keys())
            rts.sort()
            for samp in range(len(result)):
                isoWidth = result[samp]["center m/z"] - result[samp]["lower m/z"]

                mzScan = ms1Scans[takeClosest(rts,result[samp]["rt"])]

                peaksOfInterest = [[m, i] for m, i in mzScan.items() if m >= result[samp]["center m/z"] - isoWidth - 1.3
                                   and m <= result[samp]["center m/z"] + isoWidth]
                peaksOfInterest.sort(key=lambda x: x[0])

                precursorPeaks = [x for x in peaksOfInterest if
                                  abs(x[0] - result[samp]["center m/z"]) * 1e6 / result[samp]["center m/z"] <= ppmWidth]
                chimericPeaks = [x for x in peaksOfInterest if
                                 abs(x[0] - result[samp]["center m/z"]) * 1e6 / result[samp]["center m/z"] > ppmWidth and
                                 x[0] >= result[samp]["center m/z"] - isoWidth]
                if len(chimericPeaks) > 0 and len(precursorPeaks) > 0:
                    result[samp]["percentContamination"] = min([1, max([0, simps([x[1] for x in chimericPeaks],
                                                                                  [x[0] for x in chimericPeaks],
                                                                                  even="avg") / (
                                                                                 simps([x[1] for x in chimericPeaks],
                                                                                       [x[0] for x in chimericPeaks],
                                                                                       even="avg") + simps(
                                                                             [x[1] for x in precursorPeaks],
                                                                             [x[0] for x in precursorPeaks],
                                                                             even="avg"))]), ])
                elif len(chimericPeaks) > 0 and any(x[1] > 1e-2 for x in chimericPeaks):
                    result[samp]["percentContamination"] = 1.0
                else:
                    result[samp]["percentContamination"] = 0.0
                result[samp]["fragments"] = [x[0] for x in peaksOfInterest if x[1] > 1e-6]
                result[samp]["ms1"] = [x for x in peaksOfInterest if x[0] >= result[samp]["center m/z"] - isoWidth]

#             if delete:
#                 os.remove(filename)
#                 os.remove("junk.txt")
        return result,ms1Scans
   #
   # except:
   #      print(filename + " does not exist or is ill-formatted")
   #      return -1,-1

def clusterSpectraByMzs(mzs,ids,mzTol = 5):

    ppmWindow = lambda m: mzTol * 1e-6 * m

    uniqueMzs = list(set(mzs))
    absUniqueMzs = []
    for x in range(len(uniqueMzs)):
        win = ppmWindow(uniqueMzs[x])
        good = True
        for x2 in absUniqueMzs:
            if abs(x2 - uniqueMzs[x]) <= win:
                good = False
                break
        if good:
            absUniqueMzs.append(np.round(uniqueMzs[x],6))


    prelimClusters = {mz:[] for mz in absUniqueMzs}

    for mz in absUniqueMzs:
        win = ppmWindow(mz)
        toRemove = []
        for m,id,i in zip(mzs,ids,range(len(ids))):
            if abs(mz-m) <= win:
                prelimClusters[mz].append([m,id])
                toRemove.append(i)

        mzs = [mzs[x] for x in range(len(ids)) if x not in toRemove]
        ids = [ids[x] for x in range(len(ids)) if x not in toRemove]

    return prelimClusters
def flatten(l):
    if len(l) > 0 and type(l[0]) == type(l):
        return [item for sublist in l for item in sublist]
    else:
        return l

def extractChromatogram(mz,ms1,mError = 5):
    allRt = list(ms1.keys())
    allRt.sort()
    chromatogram = {}
    allMzList = list()
    for s in ms1:
        allMzList += [x for x in ms1[s] if ms1[s][x] > 0]
    allMzList = np.array(list(set(allMzList)))

    #plt.figure()
    
    mzOI = np.extract(np.abs((10**6)*(allMzList-mz))/mz < mError,allMzList)
    #print("foundmzOI")
    getInt = lambda rt: np.sum([ms1[rt][x] for x in mzOI if x in ms1[rt]])
    #maxInt,apex = findMax(mzOI,ms1,rtRep,allRt)
    # maxInt = np.max(list(tempMs1.values()))
    #print("XCR Done")
    #r = allRt.index(rtRep)

    tempMs1 = []
    for rt in allRt:
        tempMs1.append([rt,getInt(rt)])
    
    
    plt.figure()
    plt.plot([x[0] for x in tempMs1],[x[1] for x in tempMs1])
    plt.xlabel("rt")
    plt.ylabel("intensity")
    plt.title(str(mz))

    tempMs1.sort(key = lambda x: x[1],reverse=True)
    rtOfApex = tempMs1[0][0]
    return rtOfApex

    


# In[4]:



filename = "Metabolite_AcCoA_Infusion_neg_NCE_opt_10uL.raw"
filename = "Metabolite_MS-Mix_1_Infusion_neg_NCE_opt_01.raw"
#filename = "Metabolite_Hex6P_Infusion_neg_NCE_opt_10uL.raw"
#filename = "Metabolite_Citrate_Infusion_neg_NCE_opt_10uL.raw"
filename = "Citrate_ddMS2_pos.raw"
filename = "Hypotaurine_ddMS2_pos.raw"
filename = "Glutamine_ddMS2_pos.raw"

ms2,ms1 = readRawDataFile("../20191212_FIA_IROA_MSMLS_Plate1/"+filename,1000,[1],True,)
polarity = ms2[0]["mode"]


# In[10]:


metName = filename.split("_")[0]
metName = metName.upper()
rtWindow = .3
#peakFileName = "AA-Mix_m_z_Rt.csv"
peakFileName = "plate1_IROA_tab.csv"

massErr = 10
rtErr = .5
data = [x.rstrip().split(",") for x in open(peakFileName,"r").readlines()[1:]]
if polarity == "Positive":
    adductIndex = 4
else:
    adductIndex = 3
compoundsOI = {x[0]:{"m/z":float(x[adductIndex]),"rt":-1} for x in data}
maxSim = [[k,similar(k,metName)] for k in compoundsOI]
maxSim.sort(key=lambda x: x[1],reverse = True)
compoundsOI = {k[0]:compoundsOI[k[0]] for k in maxSim[:1]}
#print(compoundsOI)

clusters = {(val["m/z"],key):[] for key,val in compoundsOI.items()}
for samp in ms2:
    #print(samp["rt"])
    for name,comp in compoundsOI.items():
        if (1e6)*abs(samp["center m/z"]-comp["m/z"])/comp["m/z"] < massErr:
            if comp["rt"] == -1:
                comp["rt"] = extractChromatogram(comp["m/z"],ms1,10)
            if abs(samp["rt"]-comp["rt"]) < rtWindow:
                clusters[(comp["m/z"],name)].append(samp["id"])
    #break
#clusters = clusterSpectraByMzs([x["center m/z"] for x in ms2],[x["id"] for x in ms2],5)


# In[ ]:





# In[11]:



print(len(clusters))
print(len(ms2))


# In[12]:


results = {}
lowestProduct = -3
for clu in clusters:
    cl = clu[0]
    name = clu[1]
    ids = [x for x in clusters[clu]]
    ms2OI = [x for x in ms2 if x["id"] in ids]# and x["percentContamination"] < 0.3]
    #print(len(ms2OI))
    #cont = [x["percentContamination"] for x in ms2OI]
    if len(ms2OI) > 0:
        polarity = ms2OI[0]["mode"]
        CEs = list(set([x["NCE"] for x in ms2OI]))
        CEs.sort()
        allFrags = list(set(flatten([list(x["spectra"][0].keys()) for x in ms2OI])))
        allFrags.sort(key=lambda x: abs(x-cl))
        precursor =  [x for x in allFrags[:3] if abs(x-cl) < .15]
        allFrags = [f for f in allFrags if f <= cl+.1+lowestProduct]
        fragInt = {f:0 for f in allFrags}
        for f in allFrags:
            for s in ms2OI:
                if f in s["spectra"][0]:
                    fragInt[f] += s["spectra"][0][f]/(10**s["signal"])
        precursorFragInt = {f:0 for f in precursor}
        for f in precursorFragInt:
            for s in ms2OI:
                if f in s["spectra"][0]:
                    precursorFragInt[f] += s["spectra"][0][f]/(10**s["signal"])
        precursor.sort(key=lambda x: precursorFragInt[x],reverse=True)
        precursor = precursor[0]
        allFrags.sort(key=lambda x: fragInt[x],reverse=True)
        allFrags = allFrags[:5]
        allFrags = [precursor] + allFrags
        #precursorFrag = [x for x in range(len(allFrags)) if abs(allFrags[x]-cl) < 0.1]

        gotPrecursor = True
#         if len(precursorFrag) > 0:
#             precursorFrag = precursorFrag[0]
#             if precursorFrag != len(allFrags) - 1:
#                 allFrags = [allFrags[precursorFrag]] + allFrags[:precursorFrag] + allFrags[precursorFrag+1:]
#             else:
#                 allFrags = [allFrags[precursorFrag]] + allFrags[:precursorFrag]
#         else:
#             allFrags = allFrags[:5]
#             gotPrecursor = False


        cePartioned = [[y for y in ms2OI if y["NCE"]==c] for c in CEs]
        ceMerged = []
        for ce in cePartioned:
            mergedSpec = dict()
            for spec in ce:
                for frag,inten in spec["spectra"][0].items():
                    if frag not in mergedSpec:
                        mergedSpec[frag] = inten/len(ce)
                    else:
                        mergedSpec[frag] += inten/len(ce)
            ceMerged.append(mergedSpec)
            
        breakdownCurve = {frag:{c:0 for c in CEs} for frag in allFrags}

#         rtRange = []
#         for c,CE in zip(cePartioned,CEs):
#             c.sort(key=lambda x: x["signal"],reverse=True)
#             rtRange.append(c[0]["rt"])

        for c,CE in zip(ceMerged,CEs):
            #c.sort(key=lambda x: x["signal"],reverse=True)
            for frag in allFrags:
                if frag in c:
                    breakdownCurve[frag][CE] += c[frag]
        if gotPrecursor:
            otherFrag = allFrags[1:]
            otherFrag.sort(key=lambda x: np.max(list(breakdownCurve[x].values())),reverse=True)
            allFrags = [allFrags[0]] + otherFrag
        else:
            allFrags.sort(key=lambda x: np.max(list(breakdownCurve[x].values())),reverse=True)
        allFrags = [np.round(x,1) for x in allFrags]

        results[cl] = {"breakdown":breakdownCurve,"CEs":CEs,"polarity":polarity,"precursor":gotPrecursor,"frag":allFrags,"name":name}
    


# In[13]:



pp = PdfPages(filename.split(".")[0] + "_breakdown.pdf")
for cl in results:
    breakdownCurve = results[cl]["breakdown"]
    #rtRange = results[cl]["rtRange"]
    CEs = results[cl]["CEs"]
    plttt = plt.figure()
    allFrags = results[cl]["frag"]
    name = results[cl]["name"]

    plt.title(name  + " m/z = " + str(np.round(cl,2)))# + " | RT Range: [" + str(np.round(min(rtRange),2)) + "," + str(np.round(max(rtRange),2)) + "]")
    offset = 0
    index = 1
    offsets = []
    for f in allFrags:
        #plt.plot([float(c) for c in CEs],[breakdownCurve[f][c] for c in CEs],label=str(np.round(f,2)))
        if offset == 0 and gotPrecursor:
            label = "Precursor: "
        else:
            label = "Fragment " + str(index) + ": "
            index += 1
        plt.bar([x+offset for x in range(len(CEs))],[np.log(breakdownCurve[f][c]) for c in CEs],label=label + str(np.round(f,2)))
        offsets.append(offset)
        offset += len(CEs) + 1
    xtick = flatten([x+offset for x in range(len(CEs)) for offset in offsets])
    labels = flatten([str(ce) for ce in CEs for offset in offsets])
    plt.xticks(xtick,labels,rotation = 45,size = 7)
    plt.xlabel("NCE")
    plt.ylabel("Log(Intensity)")
    plt.legend()
    plt.tight_layout()
    
    pp.savefig(plttt)
pp.close()


# In[25]:



outfile = open(filename.split(".")[0] + "_breakdown.csv","w")
for cl in results:
    breakdownCurve = results[cl]["breakdown"]
    #rtRange = results[cl]["rtRange"]
    CEs = results[cl]["CEs"]
    gotPrecursor = results[cl]["precursor"]
    allFrags = results[cl]["frag"]
    name = results[cl]["name"]
    polarity = results[cl]["polarity"]
    
    outfile.write(name+",precursor: " + str(cl))# + ","  + "RT Range: [" + str(np.round(min(rtRange),2)) + "-" + str(np.round(max(rtRange),2)) + "]\n")
    #outfile.write(",")
    if gotPrecursor: 
        outfile.write(",Precursor " + "(" + str(allFrags[0])+")")
        [outfile.write(",Fragment " + str(ind+1) + "(" + str(f)+")") for f,ind in zip(allFrags[1:],range(len(allFrags)-1))]
    else:
        [outfile.write(",Fragment " + str(ind+1) + "(" + str(f)+")") for f,ind in zip(allFrags,range(len(allFrags)))]

    outfile.write("\n")
    
    
    for CE in CEs:
        outfile.write(str(CE))
        for frag in allFrags:
            outfile.write(","+str(breakdownCurve[frag][CE]))
        outfile.write("\n")
    
    bestNCE = dict()
    bestCEs = dict()
    for frag in allFrags:
        m = -1
        bestCE = -1
        for CE in CEs:
            if m <= breakdownCurve[frag][CE]:
                bestCE = CE
                m = breakdownCurve[frag][CE]
        bestNCE[frag] = m
        bestCEs[frag] = bestCE
        
    outfile.write("\n\n")
    outfile.write("m/z")
    [outfile.write(","+str(frag)) for frag in allFrags]
    
    outfile.write("\nOptimal NCE")
    [outfile.write(","+str(bestCEs[frag])) for frag in allFrags]
    
    outfile.write("\nOptimal Intensity")
    [outfile.write(","+str(bestNCE[frag])) for frag in allFrags]
    outfile.write("\n\n\n")
outfile.close()


# In[26]:



outfile = open(filename.split(".")[0] + "_SRM.csv","w")
outfile.write("Compound,Retention Time (min),Polarity,Precursor (m/z),Product (m/z),Collision Energy (V),Intensity\n")
for cl in results:
    breakdownCurve = results[cl]["breakdown"]
    rtRange = -1#results[cl]["rtRange"]
    CEs = results[cl]["CEs"]
    gotPrecursor = results[cl]["precursor"]
    allFrags = results[cl]["frag"]
    polarity = results[cl]["polarity"]

    if gotPrecursor:
        allFrags = allFrags[1:]
    name = results[cl]["name"]
    bestNCE = dict()
    bestCEs = dict()
    for frag in allFrags:
        m = -1
        bestCE = -1
        for CE in CEs:
            if m <= breakdownCurve[frag][CE]:
                bestCE = CE
                m = breakdownCurve[frag][CE]
        bestNCE[frag] = m
        bestCEs[frag] = bestCE
    
    for frag in allFrags:
        outfile.write(name+","+str(np.mean(rtRange))+","+polarity+","+str(cl)+","+str(frag)+","+str(bestCEs[frag])+","+str(bestNCE[frag]))


        outfile.write("\n")
outfile.close()


# In[ ]:





# In[ ]:




