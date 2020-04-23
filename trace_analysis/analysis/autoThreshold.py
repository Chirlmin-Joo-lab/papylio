# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:25:17 2016

@author: ikatechis
"""

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import os
from scipy import signal
from trace_analysis.analysis.decorators import timer


#@timer
def stepfinder(trace, threshold=100, max_steps=20):

    start_frames = []
    if trace[0] > threshold and trace[1] > threshold:
        start_frames.append(0)

    stop_frames = []
    if trace[-1] > threshold and trace[-2] > threshold:
        stop_frames.append(trace.size - 1)

    i = 0
    while i < trace.size -2:
        dif1 = trace[i+1] - trace[i]
        dif2 = trace[i+2] - trace[i]

        if ((dif1 > threshold and
            dif2 > threshold and
            trace[i] < threshold) or
            (i==0 and start_frames) ): # this will not catch stoichiometry of 2
            # start_frames.append(i+1)
            for j in range(i+2, trace.size - 2):  # start 2 positions after the step start until the length of the original trace
                dif1 = trace[j+1] - trace[j]
                dif2 = trace[j+2] - trace[j]

                if dif1 < -threshold and dif2 < -threshold\
                            and trace[j+2] < threshold:
                    start_frames.append(i+1)
                    stop_frames.append(j+1)
                    i = j+1
                    break
        else:
            i +=1
        # i += 1 # start the next loop from the last stop frame

    if len(start_frames) != len(stop_frames):  # sometimes 2 consecutive frames both satisfy the threshold condition for very big jumps
        start_temp = np.copy(start_frames)
        stop_temp = np.copy(stop_frames)
        for i, start in enumerate(start_temp):
            if  start_temp[i] - start_temp[i-1] == 1:
#                print('Found two consecutive start frames')

                start_frames.remove(start)
        for i, stop in enumerate(stop_temp):
            if  stop_temp[i] - stop_temp[i-1] == 1:
                print('Found two consecutive stop frames')
                stop_frames.remove(stop)

    if len(start_frames) != len(stop_frames): # if the problem remains
        print ("something is wrong")
        print ("start frames: "+str(start_frames))
        print ("stop frames: "+str(stop_frames))
        return {"frames": np.array([])}

    elif len(start_frames) > max_steps:
        print ("Found more steps than the limit of "+str(max_steps))
        return {"frames": np.array([])}
    else:
        print ("steps found: " + str(len(start_frames+stop_frames)))
        if len(start_frames+stop_frames) % 2 > 0:
            print('odd number of steps found. Result discarded.')
        print ("start frames: "+str(start_frames))
        print ("stop frames: "+str(stop_frames))
        res={ "frames": np.array(start_frames+stop_frames),
             "threshold": threshold}
        return res


def plot_steps(trace="red", exposure_time=0.1, steps={},
               name="molecule_0", display_plot=False,
               save_plot=True, save_folder="./saved_plots"):

    Nframes = trace.size
    time = np.linspace(0, Nframes*exposure_time, Nframes)
    if not display_plot:
        plt.ioff()
    plt.figure(name, figsize=(10, 4))


    if "start_times" in steps.keys():
        start_times = steps["start_times"]
        stop_times = steps["stop_times"]
        for start, stop in zip(start_times, stop_times):
            plt.axvline(start, c="green", lw=2, ls="-.")
            plt.axvline(stop, c="red", lw=1, ls="--")
    plt.plot(time, trace, "k", lw=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.xlim((min(time), max(time)))
    plt.tight_layout()
    if save_plot:
        if save_folder not in listdir("."):
            os.mkdir(save_folder)
        plt.savefig("./"+save_folder+"/"+name)
        if not display_plot:
            plt.close()
            plt.pause(0.001)
    plt.ion()



#read_selected = True
#thresholds = [100]
#differences = ["2"]
#
#
#os.chdir("G:/TIRF Data/collected")
#
#if read_selected:
#    dirs = []
#    for path, subdirs, files in os.walk("."):
#
#        for name in subdirs:
#            if "selected" in name:
#                dirs.append(os.path.join(path, name))
#
#if not read_selected:
#    dirs = []
#    for d in os.listdir("."):
#        dirs.append(os.path.abspath(d))
#
#for d in dirs:
#    os.chdir(d)
#    for thres in thresholds:
#        for Ndif in differences:
#            threshold = thres
#            if read_selected:
#                os.chdir("..")
#                name = d[d.find("hel"):]
#                dwell_times = analyzer(name, threshold, read_selected=read_selected, Ndif=Ndif)
#                plt.figure("Dwell-times histogram - "+name)
#                plt.hist(dwell_times, bins=30)
#                plt.savefig("./dwell_times_threshold_"+str(threshold)+"_selected_"+Ndif+"dif_{}".format(name))
#                plt.close()
#
#            if not read_selected:
#                names = [n[:n.find(".traces")] for n in os.listdir(".") if ".traces" in n]
#                for name in names:
#                    dwell_times = analyzer(name, threshold, read_selected=read_selected, Ndif=Ndif)
#
#                    plt.figure("Dwell-times histogram - "+name)
#                    plt.hist(dwell_times, bins=30)
#                    plt.savefig("./dwell_times_threshold_"+str(threshold)+"_selected_"+Ndif+"dif_{}".format(name))
#                    plt.close()
#    os.chdir("..")


#
#    for i in range(len(acceptor)):
#    plt.figure(data["names"][i])
#    plt.plot(time, acceptor[i])
#    plt.show()


#trace = acceptor[0]
#plt.figure("Trace")
#plt.plot(trace)
#trace_filt = signal.savgol_filter(trace, 11, 2)
#plt.plot(trace_filt)
#plt.figure()
#plt.hist(trace, bins=100)

#def check_differences(trace, threshold, Ndif=4):
#    for i, t in enumerate(trace):
#        if i <= len(trace) - Ndif+1:
#            dif = [trace[i+j] - t for j in range(1,Ndif+1)]
#            b_up = []
#            b_down = []
#            for d in dif:
#                if d > threshold and t < threshold:
#                    b_up.append(True)
#                elif d < -threshold:
#                    b.append(False)
#
#            if all(b):
#                print "step at "+str(i)
#                print dif
#                print b