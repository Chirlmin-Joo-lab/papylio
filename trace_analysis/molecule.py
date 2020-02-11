import numpy as np #scientific computing with Python
import matplotlib.pyplot as plt #Provides a MATLAB-like plotting framework
from trace_analysis.analysis.autoThreshold import stepfinder


class Molecule:

    def __init__(self, file):
        self.file = file
        self.index = None
        self._coordinates = None
        self.intensity = None
        self.frames_per_second = int(self.file.experiment.configuration['frames']['frames_per_second'])
        self.total_frames= int(self.file.experiment.configuration['frames']['total_frames'])
        self.tracestart = int(0)
        self.traceend = len(self.file.time)/self.frames_per_second-1/self.frames_per_second
        self.isSelected = False
        self.time=np.linspace(self.tracestart, self.traceend, (self.traceend - self.tracestart) * self.frames_per_second + 1)
        self.steps = None  #Defined in other classes as: pd.DataFrame(columns=['frame', 'trace', 'state', 'method','thres'])
        self.kon_boolean = None  # 3x3 matrix that is indicates whether the kon will be calculated from the beginning, in-between molecules or for the end only

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = np.atleast_2d(coordinates)

    def I(self, emission, Ioff=0):
        return (self.intensity[emission, :] - Ioff - self.file.background[emission])[int(self.tracestart*self.frames_per_second):int(self.traceend*self.frames_per_second+int(1))]

    def E(self, Imin=0, alpha=0, Iroff=0, Igoff=0):
        red = np.copy(self.I(1, Ioff=Iroff))
        green = self.I(0, Ioff=Igoff)
        np.putmask(red, red < Imin, 0)  # the mask makes all elements of acceptor that are below the Imin zero, for E caclulation
        E =  (red - alpha*green) / (green + red - alpha*green)
        E = np.nan_to_num(E)  # correct for divide with zero = None values
        return E

    def plot(self, figure = None):
        if not figure: figure = plt.gcf() # Or possibly e.g. plt.figure('Trace plot')
        figure.clf()
        if len(self.file.experiment.pairs) > 0:
            axis_I = figure.add_subplot(211)
        else:
            axis_I = figure.gca()

        axis_I.set_xlabel('Time (s)')
        axis_I.set_ylabel('Intensity (a.u.)')
        axis_I.set_ylim(0, 500)
        for i, colour in enumerate(self.file.experiment.colours):
            axis_I.plot(self.time, self.I(i), colour)

        if len(self.file.experiment.pairs) > 0:
            axis_E = figure.add_subplot(212, sharex = axis_I)
            axis_E.set_xlabel('Time (s)')
            axis_E.set_ylabel('FRET (-)')
            axis_E.set_ylim(0,1)
            for i, pair in enumerate(self.file.experiment.pairs):
                axis_E.plot(self.time, self.E())
    @property  # this is just for the stepfinder to be called through Molecule. Maybe not needed
    def find_steps(self):
        return stepfinder
        #plt.show()
#MD190104: why not add a subplot with FRET here as well, to match with display Matlab?

#    @property
#    def find_steps(self):
#        return stepfinder