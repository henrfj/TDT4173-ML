import ipywidgets
from IPython.display import display
from threading import Thread
import pandas as pd
import time
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf


def preProcessData(metaData : dict, data : pd.DataFrame):
    for index, row in data.iterrows():
        pass

    
def stopLoop(stop):
    input()
    stop[0]=True

def rangeProgressBar(start, end=None, cancellable=False):
    def pad(s,num,pad):
        l = max(num-len(s),0)
        return s + "".join([pad for i in range(l)])
    
    if end is None:
        end = start
        start = 0
    startTime = time.time()
    nextUpdateBarTime = startTime
    
    stop = [False]
    if cancellable:
        Thread(target=stopLoop, args=(stop,)).start()
    
    orientation = 0
    label = ipywidgets.HTML()
    #button = ipywidgets.Button(description="Click Me!")
    #hBox = ipywidgets.HBox([label,button])
    #display(hBox)
    display(label)
    i = 0
    while i < end:
        
        if i == end-1 or time.time() >= nextUpdateBarTime:
            nextUpdateBarTime = time.time() + 0.5 # + 1/2 second.
            
            if i == end-1:
                index = end
            else:
                index = i
            progress = index/end
            
            text = "<p style=\"font-family:'Lucida Console', monospace\">["
            barLength = 30
            barsDone = int(barLength*progress)
            text += "".join(["|" for i in range(barsDone)])
            if barLength-barsDone > 0:
                orientation = (orientation+1) % 4
                text += ["▄","█","▀","█"][orientation]
                #text += ["▌","█","▐","█"][orientation]
                text += "".join([":" for i in range(barLength-barsDone-1)])
            text += "]&nbsp;&nbsp;"
            
            text += pad(str(int(progress*100)),3,'&nbsp') + "%&nbsp;&nbsp;"
            
            secondsPast = (time.time()-startTime)
            expectedFinish = secondsPast / ((1 if progress == 0 else progress))
            timeLeft = expectedFinish - secondsPast
            text += "Elapsed: {}:{}:{}&nbsp;&nbsp;Total: {}:{}:{}&nbsp;&nbsp;Left: {}:{}:{}".format(
                pad(str(int(secondsPast/60/60)),2,"&nbsp;"),
                pad(str(int(secondsPast/60)%60),2,"&nbsp;"),
                pad(str(int(secondsPast)%60),2,"&nbsp;"),
                pad(str(int(expectedFinish/60/60)),2,"&nbsp;"),
                pad(str(int(expectedFinish/60)%60),2,"&nbsp;"),
                pad(str(int(expectedFinish)%60),2,"&nbsp;"),
                pad(str(int(timeLeft/60/60)),2,"&nbsp;"),
                pad(str(int(timeLeft/60)%60),2,"&nbsp;"),
                pad(str(int(timeLeft)%60),2,"&nbsp;")
            )
            
            label.value = text + "</p>"
        if stop[0]:
            return
        yield i
        i += 1

def multiGraph(graphs, titles, yMax=None, yMin=0):
    if type(graphs[0]) != list:
        graphs = [graphs]
    
    maxLength = max([len(v) for v in graphs])
    maxValue = max([max(v) for v in graphs])
    x = np.arange(0, maxLength, 1)
    fig, axs = plt.subplots(1, figsize=(15, 15))  # Create a figure and an axes.
    
    currentAxes = plt.gca()
    if yMax is not None:
        currentAxes.set_ylim([yMin,yMax])
        
    
    def stepSize(x):
        density = 20
        thenth = 10**(math.floor(math.log(x/density,10)))
        if x/thenth > density*10/2:
            thenth = thenth*5
        if x/thenth > density*10/4:
            thenth = thenth*2
        return thenth
    
    axs.set_xticks(np.arange(0, maxLength, stepSize(maxLength)))
    mm = min(maxValue, yMax) if yMax is not None else maxValue
    axs.set_yticks(np.arange(0, mm, stepSize(mm)))
    axs.tick_params(axis='both', which='major', labelsize=15)
    plt.xticks(rotation=90)
    axs.grid()
    
    for i in range(len(graphs)):
        g = graphs[i]
        t = titles[i]
        axs.plot(x, g, label=t)  # Plot some data on the axes.
    axs.set_xlabel('Epochs', fontsize=20)  # Add an x-label to the axes.
    axs.set_ylabel('Loss', fontsize=20)  # Add a y-label to the axes.
    axs.set_title("")  # Add a title to the axes.
    axs.legend(titles, fontsize=10)  # Add a legend.
    