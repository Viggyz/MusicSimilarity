import os, sys, errno
import re
import argparse
from time import time
import multiprocessing,glob

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa.display
import multiprocessing

precalculated_files = "precalculated_files_mel/fma_single/"

def plotData(file):
    name = Path(file).stem
    if Path(precalculated_files+name+'.png').is_file():
        # print("Exists",name)
        return -1
    melspectrogram = np.load("precalculated_files_mel\\fma_numpy\\%s.npy" % name)
    power_to_db = librosa.power_to_db(melspectrogram, ref=np.max) #Run with vmin and vmax
    librosa.display.specshow(power_to_db, x_axis='s', y_axis='mel', cmap='gray_r',vmin=0,vmax=255)
    plt.axis('off')
    plt.savefig(precalculated_files+name,bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
    # print("Saved",name )
    # Clear figure so that the next plot this worker makes will not contain
    # data from previous plots

    return name   

if __name__=="__main__":
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print("Making plots using %d processors..." %
          (multiprocessing.cpu_count()))
    
    tasks = []
    for file in glob.iglob("precalculated_files_mel/fma_numpy/*"):
        tasks.append((file,))
    
    # print(tasks[1:10])
    
    results = [pool.apply_async(plotData, t) for t in tasks]

    # Process results
    for result in results:
        name = result.get()
        print("Exists" if name == -1 else "Saved: %s" % name)

    pool.close()
    pool.join()
