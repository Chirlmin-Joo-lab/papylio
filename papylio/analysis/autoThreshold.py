import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import os


def stepfinder(trace, threshold=100, max_steps=20):

    start_frames = []
    if trace[0] > threshold and trace[1] > threshold:
        start_frames.append(0)

    stop_frames = []
    if trace[-1] > threshold and trace[-2] > threshold:
        stop_frames.append(trace.size - 1)

    i = 0
    while i < trace.size -2:
        print(i)
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