# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:12:58 2019

@author: iason
"""
if __name__ == '__main__':
    import sys
    from pathlib import Path
    p = Path(__file__).parents[2]  # include two levels up directory where the trace_analysis package lies
    sys.path.insert(0, str(p))

import numpy as np

from trace_analysis import Experiment
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.widgets
import seaborn as sns
from pathlib import Path, WindowsPath
import functools
#plt.rcParams['toolbar'] = 'toolmanager'
# from matplotlib.backend_tools import ToolBase


class InteractivePlot(object):
    def __init__(self, molecules, canvas=None):

        self.molecules = molecules
        self.mol_indx = 6  #From which molecule to start the analysis
        self.canvas = canvas


    @property
    def file(self):
        return self.molecules[self.mol_indx].file
    @property
    def time(self):
        return self.molecules[self.mol_indx].file.time

    def plot_initialize(self):
        plt.ion() # Possibly we should fix this in another way [IS: 18-11-2019]

        sns.set(style="dark")
        sns.set_color_codes()
        plt.style.use('dark_background')
        if self.canvas is None:
            self.fig, self.axes = plt.subplots(2, 1, sharex=True, figsize=(10,5))
        else:
            self.fig = self.canvas.figure
            self.axes = self.fig.subplots(2, 1, sharex=True)


        self.axes[0].set_ylim((-50, 1500))  # Det default intensity limits
        self.axes[0].set_ylabel("intensity (a.u)\n")
        self.axes[1].set_ylim((-0.1,1.1))  # Set default fret limits
        self.axes[1].set_xlabel("time (s)")
        self.axes[1].set_ylabel("FRET\n")
        self.axes[1].set_xlim((-10, self.time[-1]+10))  # Set default x limits dependent on measurement time

        plt.subplots_adjust(bottom=0.23)

        # Create the axes for the widgets
        self.rax = plt.axes([0.91, 0.71, 0.08, 0.2])


        self.axcheckfret = plt.axes([0.91, 0.35, 0.08, 0.08])
        self.axcorred = plt.axes([0.95, 0.6, 0.028, 0.06])
        self.axcorgreen = plt.axes([0.95, 0.53, 0.028, 0.06])
        self.axcorrfretI = plt.axes([0.95, 0.3, 0.028, 0.06])
        self.axthrsliders = [plt.axes([0.26, 0.072, 0.10, 0.03]),
                             plt.axes([0.26, 0.033, 0.10, 0.03])]
        # Create the buttons
        self.axthresb = plt.axes([0.1, 0.03, 0.13, 0.062])  # Button to calculate dwell times by thresholding
        self.axrejb = plt.axes([0.41, 0.03, 0.07, 0.062])   # Button to reject calculated dwell times by thresholding
        self.axclearb = plt.axes([0.51, 0.03, 0.11, 0.062]) # Button to clear the clicked points (clears vlines)
        self.axthrowb = plt.axes([0.64, 0.03, 0.11, 0.062]) # Button to throw away already calculated dwell times and de-select molecule
        self.axconclb = plt.axes([0.77, 0.03, 0.15, 0.062]) # Button to conlcude analysis by saving all the calculated steps and metadata

        self.axnextb = plt.axes([0.162, 0.90, 0.065, 0.062])  # Buttons to cycle through molecules
        self.axprevb = plt.axes([0.075, 0.90, 0.08, 0.062])

        self.axsavetraceb = plt.axes([0.26, 0.90, 0.065, 0.062])

        self.axgotomolecule = plt.axes([0.50, 0.89, 0.05, 0.062])
        self.axplotselected = plt.axes([0.63, 0.90, 0.065, 0.062])
        self.axsaveselectedb = plt.axes([0.77, 0.90, 0.065, 0.062])

        self.axtotal = plt.axes([0.865, 0.90, 0.065,  0.062])
        [ax.set_frame_on(False) for ax in self.fig.get_axes()[2:]]


        #  Radiobutton to select red or green
        self.radio = matplotlib.widgets.RadioButtons(self.rax, ("red", "green", 'total'))
        self.radio.circles[0].set_color("r")
        for circle in self.radio.circles: # adjust radius here. The default is 0.05
            circle.set_radius(0.07)
        self.radio.on_clicked(self.radio_manage)

        #  Connect clicking to draw lines class
        self.draw = Draw_lines(self.fig, self.radio)

        # Create the control buttons
        bp = {'color': 'black', 'hovercolor': 'gray'}
        self.bauto = matplotlib.widgets.Button(self.axthresb,'autoThreshold' , **bp)
        self.bauto.on_clicked(self.autoThreshold_plot)
        self.brejauto = matplotlib.widgets.Button(self.axrejb,'reject' , **bp)
        self.brejauto.on_clicked(self.auto_reject)
        self.bclear = matplotlib.widgets.Button(self.axclearb,'clear clicks' , **bp)
        self.bclear.on_clicked(self.draw.clear_all)
        self.bthrow = matplotlib.widgets.Button(self.axthrowb,'throw away' , **bp)
        self.bthrow.on_clicked(self.throw_away)
        self.bconcl = matplotlib.widgets.Button(self.axconclb,'conclude analysis' , **bp)
        self.bconcl.on_clicked(self.conclude_analysis)
        self.bnext = matplotlib.widgets.Button(self.axnextb,'Next' , **bp)
        self.bnext.on_clicked(self.save_molecule)
        self.bprev = matplotlib.widgets.Button(self.axprevb,'Previous' , **bp)
        self.bprev.on_clicked(self.save_molecule)

        self.bsavetrace = matplotlib.widgets.Button(self.axsavetraceb,'save trace' , **bp)
        self.bsavetrace.on_clicked(self.save_trace)

        # an entry box to go to desired molecule
        gotodict = {'initial': str(1), 'color':'k', 'hovercolor': 'k', 'label_pad':.2}
        self.gotoentry = matplotlib.widgets.TextBox(self.axgotomolecule,
                                                    'Go To:', **gotodict)

        # a checkbutton to whether plot only the selected molecules
        self.checkbplotselected = matplotlib.widgets.CheckButtons(self.axplotselected,
                                                                  ["plot selected"],
                                                                  actives=[False])

        # A button to save selected molecules
        self.bsaveselected = matplotlib.widgets.Button(self.axsaveselectedb,'save selected' , **bp)
        self.bsaveselected.on_clicked(self.save_selected)

        #  A checkbutton for whether to display the total intensity
        self.checkbtotal = matplotlib.widgets.CheckButtons(self.axtotal, ["show total"],
                                                          actives=[False])

        #  A checkbutton for fret autothreshold dwell-time calculation
        self.checkbfret = matplotlib.widgets.CheckButtons(self.axcheckfret, ["E fret"],
                                                          actives=[False])

        #  Remove the x lines from the checkbuttons
        for chbutton in [self.checkbplotselected, self.checkbtotal, self.checkbfret]:
            chbutton.rectangles[0].set_color("black")
            chbutton.rectangles[0].set_height( 0.2)
            [line.remove() for line in chbutton.lines[0]]

        #  Entryboxes for offset corrections
        corrdict = {'initial': str(0), 'color':'k', 'hovercolor': 'k', 'label_pad':.2}
        corrlabels = [r'$I_{R_{off}}$', r'$I_{G_{off}}$', r'$I_{min}$']
        corraxes = [self.axcorred, self.axcorgreen, self.axcorrfretI]
        self.correntries = [matplotlib.widgets.TextBox(ax, label, **corrdict)
                            for ax, label in zip(corraxes, corrlabels)]

        #  Sliders for assigning the threshold
        self.thrsliders = []
        self.thrsliders.append(matplotlib.widgets.Slider(self.axthrsliders[0],
                                                         label=r"$I_R$", valmin=0,
                                                         valmax=200, valinit=50,
                                                         valfmt="%i", color="r"))
        self.thrsliders.append(matplotlib.widgets.Slider(self.axthrsliders[1],
                                                         label=r"$E$", valmin=0,
                                                         valfmt="%.2f", valinit=0.5,
                                                         color="b", valmax=1.0))
        [slider.vline.remove() for slider in self.thrsliders]  # remove the default vertical lines showing the initial value

        self.connect_events_to_canvas()


    def connect_events_to_canvas(self):
        self.fig.canvas.mpl_connect('button_press_event', self.draw.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.key_bind)
        self.fig.canvas.mpl_connect('axes_leave_event', functools.partial(self.change_axis, 'leave'))
        self.fig.canvas.mpl_connect('axes_enter_event', functools.partial(self.change_axis, 'enter'))
        [entry.on_submit(lambda _: self.plot_molecule()) for entry in self.correntries]
        [slider.on_changed(functools.partial(self.change_slider, slider)) for slider in self.thrsliders]
        self.checkbplotselected.on_clicked(self.checkbutton_color)
        self.gotoentry.on_submit(self.go_to_molecule)
        self.checkbtotal.on_clicked(self.check_total)
        self.checkbfret.on_clicked(self.checkbutton_color)


    def plot_molecule(self, draw_plot=True):
        sns.set(style="dark")
        sns.set_color_codes()
        plt.style.use('dark_background')
        # clear the appropriate lines from axes first
        [ax.lines.clear() for ax in self.fig.get_axes()[:2]]
        # find the current molecule instance
        self.mol = self.molecules[self.mol_indx]
        print(f'mol. coordinates: {self.mol.coordinates}')
        self.fig.canvas.set_window_title(f'Dataset: {self.file.name}')

        # Check if molecule is selected, This will also set the title of the axis
        if self.mol.isSelected:
            self.select_molecule(toggle=False)
        else:
            self.select_molecule(toggle=False, deselect=True)
        #load saved steps if available
        self.load_from_Molecule()

        # load kon if existing or assign a False 3x3 boolean
        self.prev_mol = self.molecules[self.mol_indx - 1]
        if all(kon is None for kon in [self.mol.kon_boolean, self.prev_mol.kon_boolean]):
            self.kon = np.zeros((4,3), dtype=bool)
        elif self.mol.kon_boolean is  None:
            self.kon = np.copy(self.prev_mol.kon_boolean)  # if no kon is defined for current molecule
        else:
            self.kon = self.mol.kon_boolean

        # update the edge color from self.kon:
        self.load_edges(load_fret=True)

        self.Iroff, self.Igoff, self.Imin = [float(c.text) for c in self.correntries]

        self.red = self.mol.I(1, Ioff=self.Iroff)
        self.green = self.mol.I(0, Ioff=self.Igoff)
        self.total = self.red + self.green


        self.fret = self.mol.E(Imin=self.Imin, Iroff=self.Iroff, Igoff=self.Igoff)

        if not draw_plot:
            return

        #  if draw_plot is true
        # The order in which the traces are plotted matters later for the order of the axes lines.
        # it should always follow the order of the radio button
        self.lred = self.axes[0].plot(self.time, self.red, "r", lw=.75)[0]
        self.lgreen = self.axes[0].plot(self.time, self.green, "g", lw=.75)[0]
        self.ltotal = self.axes[0].plot(self.time, self.total, "bisque", lw=.65,
                               zorder=-1, visible=self.checkbtotal.get_status()[0])[0]

        Imax = np.max(self.red + self.green)
        self.axes[0].set_ylim((-50, Imax))

        self.axes[1].plot(self.time, self.fret, "b", lw=.75)

        # vertical lines to indicate the threshold in the two axes
        self.slidel = [ax.axhline(0, lw=1, ls=":", zorder=3, visible=False) for ax in self.axes]
        #  Creat cursor particular to the molelcule and connect it to mouse movement event
        self.cursors = []
        cursor_kws = {'useblit': True, 'color': 'white', 'ls': "--", 'lw': 1, 'alpha': 0.5}
        self.cursors.append(matplotlib.widgets.Cursor(self.axes[0], **cursor_kws))
        self.cursors.append(matplotlib.widgets.Cursor(self.axes[1], **cursor_kws))

        self.fig.canvas.draw()

    def change_axis(self, event_type, event):
        ax = event.inaxes
        if event_type == 'enter':
            if ax == self.axes[1]:
                self.fret_edge_lock = False
            elif ax in self.axthrsliders:
                indx = int(ax == self.axthrsliders[1])  # gives 0 if ax is upper (I) plot, 1 if ax is lower (E) plot
                self.slidel[indx].set_ydata(self.thrsliders[indx].val)
                self.slidel[indx].set_visible(True)
                self.fig.canvas.draw()
        elif event_type in ['leave', 'slider_change']:
            if ax == self.axes[1]:
                self.fret_edge_lock = True
            elif ax in self.axthrsliders:
                indx = int(ax == self.axthrsliders[1])  # gives 0 if ax is upper (I) plot, 1 if ax is lower (E) plot
                self.slidel[indx].set_visible(False)
                self.fig.canvas.draw_idle()

    def change_slider(self, slider, cid):
        indx = int(slider == self.thrsliders[1])  # # gives 0 if slider is I slider, 1 if slider E slider
        self.slidel[indx].set_ydata(self.thrsliders[indx].val)
        self.slidel[indx].set_visible(True)
        self.fig.canvas.draw_idle()


    def key_bind(self, event):
        k = event.key
        if k == 'a': self.autoThreshold_plot(event, find_all=False)
        if k == 'ctrl+a': self.autoThreshold_plot(event, find_all=True)
        elif k in ['w', 'e']: self.save_molecule(event, move=True)
        elif k == 'z': self.auto_reject(event)
        elif k == 'c': self.draw.clear_all(event)
        elif k in [',', '.', '/']: self.select_edge(k)
        elif k == ' ': self.select_molecule(toggle=True)
        elif k == 'r': self.radio_manage('red')
        elif k == 't': self.radio_manage('green')
        elif k == 'f':
            self.checkbfret.set_active(0)
            self.checkbutton_color('E fret')
        elif k == 'i':
            self.checkbtotal.set_active(0)
            self.check_total('Total')

        elif k == 'x': self.throw_away(event)
        elif k == 'l': self.conclude_analysis()
        elif k == '[': self.draw.select_starttrace(event)
        elif k == ']': self.draw.select_endtrace(event, self.time[-1])

        self.fig.canvas.draw_idle()


    def load_from_Molecule(self):
        if self.mol.steps is None:
            return
        else:
            s = self.mol.steps
            [self.axes[0].axvline(f, zorder=0, lw=0.65, label="saved r")
             for f in s.time[s.trace == 'red'].values]
            [self.axes[0].axvline(f, zorder=0, lw=0.65, label="saved g")
             for f in s.time[s.trace == 'green'].values]
            [self.axes[0].axvline(f, zorder=0, lw=0.65, label="saved t")
             for f in s.time[s.trace == 'total'].values]
            [self.axes[1].axvline(f, zorder=0, lw=0.65, label="saved E")
             for f in s.time[s.trace == 'E'].values]

    def select_molecule(self, toggle=True, deselect=False):
        if toggle:
            self.mol.isSelected = not self.mol.isSelected
        elif deselect:
            self.mol.isSelected = False
        else:
            self.mol.isSelected = True

        title = f'Molecule: {self.molecules.index(self.mol)+1} /{len(self.molecules)} '
        title += f'({self.mol.index + 1} /{len(self.mol.file.molecules)} in {self.mol.file.name})'
        title += '  (S)'*(self.mol.isSelected)
        rgba = matplotlib.colors.to_rgba
        c = rgba('g')*self.mol.isSelected + rgba('w')*(not self.mol.isSelected)
        self.fig.suptitle(title, color=c, fontsize=10)
        self.fig.canvas.draw_idle()

    def throw_away(self, event):
        if self.mol.steps is not None:
            lines = self.axes[0].get_lines() + self.axes[1].get_lines()
            # select only the vertical lines with corresponding labels
            lines = [l for l in lines if l.get_label().split()[0] \
                     in ['man','thres','saved']]
            # check if the lines are inside the current zoom level
            lines = [l for l in lines if \
                     l.get_xdata()[0] > self.axes[0].get_xlim()[0] and \
                         l.get_xdata()[0] < self.axes[0].get_xlim()[1]]

            # remove the corresponding data from molecule.steps
            for l in lines:
                steps = self.mol.steps.time.values
                xdat = l.get_xdata()[0]
                indx = next(i for i, _ in enumerate(steps) \
                                if np.isclose(_, xdat, 10**-4))
                self.mol.steps.drop(indx, inplace=True)
                self.mol.steps.reset_index(drop=True, inplace=True)
                print(self.mol.steps)
            # set mol.steps to None if it ends up being empty and deselect it
            if self.mol.steps.empty:
                self.mol.steps = None
                self.select_molecule(toggle=False, deselect=True)

            # remove the selected lines
            [print(f'line at {l.get_xdata()[0]} removed') for l in lines]
            [l.remove() for l in lines]

            self.fig.canvas.draw_idle()


    def save_molecule(self, event=None, move=True, draw=True):

#       Assume acceptance of automatically found and manually selected dwell times
        lines = self.axes[0].get_lines() + self.axes[1].get_lines()
        lines = [l for l in lines if l.get_label().split()[0] in ["man", "thres"]]
        self.mol.kon_boolean = self.kon
        if lines:
            if len(lines) % 2 != 0:
                print(f'Found an odd number of steps. Molecule {self.mol.index} not added')
                return
            if self.mol.steps is None:
                print('lines', len(lines))
                self.mol.steps = pd.DataFrame(columns=['time', 'trace', 'state',
                                                       'method','thres','kon',
                                                       'Iroff', 'Igoff', 'Imin'],
                                              dtype=object)
            # Molecule is automatically selected if steps are indicated
            self.mol.isSelected = True
            # turn the kon matrix into a flat string
            kon = [f'{int(i)}' for i in self.mol.kon_boolean.flatten()]
            kon = ''.join(kon)

            for l in lines:
                method = l.get_label().split()[0]
                thres = "N/A"*(method=='man') + str(self.thrsliders[0].val)*(method =='thres')

                d = {'time': max(l.get_xdata()[0], 0), 'trace': l.get_label().split()[1],
                     'state': 1, 'method': method, 'thres': thres, 'kon': kon,
                      'Iroff': self.Iroff, 'Igoff': self.Igoff, 'Imin': self.Imin,
                      'isSelected': bool(self.mol.isSelected)}
                self.mol.steps = self.mol.steps.append(d, ignore_index=True)
            self.mol.steps.drop_duplicates(inplace=True)

           #Sort the timepoints first by trace type and then in ascending order
            self.mol.steps = self.mol.steps.sort_values(by=['trace', 'time'])

        if move:
            self.move_molecule(event, draw)


    def move_molecule(self, event, draw):
        if event.inaxes == self.axnextb or event.key in ['e']:
            self.mol_indx += 1
            if self.mol_indx >= len(self.molecules) \
                or self.mol_indx <= -len(self.molecules):
                self.mol_indx = 0

        elif event.inaxes == self.axprevb or event.key in ['w']:
            self.mol_indx -= 1
            if self.mol_indx >= len(self.molecules) \
                or self.mol_indx <= -len(self.molecules):
                self.mol_indx = 0

        # if the plot_selected checkbutton is active move again until a selected
        # molecule is found
        if self.checkbplotselected.get_status()[0]:
            mol = self.molecules[self.mol_indx]
            if not mol.isSelected:
                self.move_molecule(event, draw)
            else:
                self.plot_molecule(draw_plot=draw)
        else:

            self.plot_molecule(draw_plot=draw)

    def conclude_analysis(self, event=None, save=True, filename=None):
        # Save current molecule if it was analyzed
        self.save_molecule(move=False)

        if filename is None:
            filename = f'{self.file.relativeFilePath}_steps_data.xlsx'

        if save:
            self.file.savetoExcel(filename=filename)
            # save also the selected molecules
            self.save_selected()


    def autoThreshold_plot(self, event=None, find_all=False):
        self.auto_reject()
        #  Find the steps for the checked buttons
        sel = self.radio.value_selected
        color = self.red*bool(sel == 'red') + self.green*bool(sel == 'green') \
                + self.total*bool(sel == 'total')  # Select trace data for red  or green
        # get the color indices for the current zoom level
        i_left = find_nearest(self.time, self.axes[0].get_xlim()[0])
        i_right = find_nearest(self.time, self.axes[0].get_xlim()[1])
        print(i_left, i_right)

        mask = np.zeros(color.size)
        mask[i_left: i_right] = 1
        color = np.multiply(color, mask)

        steps = self.mol.find_steps(color, threshold=self.thrsliders[0].val)
        print(steps)
        l_props = {"lw": 0.75, "zorder": 5, "label": "thres "+sel}
        [self.axes[0].axvline(s*self.file.exposure_time, **l_props) \
         for s in steps["frames"]]
        if self.checkbfret.get_status()[0]:
            fret = self.fret[i_left: i_right]
            steps = self.mol.find_steps(fret, threshold=self.thrsliders[1].val)
            l_props = {"lw": 0.75, "zorder": 5, "label": "thres E"}
            [self.axes[1].axvline(s*self.mol.file.exposure_time, **l_props) \
             for s in steps["frames"]]
        self.fig.canvas.draw()
        if find_all:
            for mol in self.molecules:
                self.autoThreshold_plot(find_all=False)
                print(f'Analyzed mol {self.mol.index} /{len(self.molecules)}')
                e = matplotlib.backend_bases.KeyEvent('key_press_event',
                                                      self.fig.canvas, 'e')
                if mol != self.molecules[-1]:
                    self.save_molecule(event=e, move=True, draw=False)
                elif mol == self.molecules[-1]:
                    self.conclude_analysis()
                    return

    def auto_reject(self, event=None):
        for ax in self.axes:
            lines = ax.get_lines()
            lines = [l for l in lines if l.get_label().split()[0] == 'thres']
            lines = [l for l in lines if \
                     l.get_xdata()[0] > self.axes[0].get_xlim()[0] and \
                         l.get_xdata()[0] < self.axes[0].get_xlim()[1]]
            [l.remove() for l in lines]
            self.fig.canvas.draw_idle()


    def radio_manage(self, label):
        def update_slider(color, label):
            s = self.thrsliders[0]
            s.poly.set_color(color); s.label.set(text=label)

        indx = [l.get_text() for l in self.radio.labels].index(label)
        color = 'r'*(label=='red') + 'g'*(label=='green') + 'bisque'*(label=='total')

        lines =  [self.lred, self.lgreen, self.ltotal]
        [l.set_zorder(-lines.index(l)) for l in lines if lines.index(l) != indx ]
#        print(indx, color)
        lines[indx].set_zorder(10)
        self.radio.circles[indx].set_color(color)
        [c.set_color('black') for c in self.radio.circles if self.radio.circles.index(c) != indx]
        update_slider(color, r"$I_G$"*(label=='green') + \
                                r"$I_R$"*(label=='red') + \
                                r"$I_T$"*(label=='total') )

        # Check the edge colors and set to white if not selected color
        # sel = self.radio.value_selected
        selcol = matplotlib.colors.to_rgba(color)
        spcol = [self.axes[0].spines[s].get_edgecolor() for s in ['left','bottom','right']]
        if selcol not in spcol:
            [self.axes[0].spines[s].set_color('white') for s in ['left','bottom','right']]

        self.load_edges()

    def load_edges(self, load_fret=False):  # loads edge color from kon array
        label = self.radio.value_selected
        indx = [l.get_text() for l in self.radio.labels].index(label)
        color = 'r'*(label=='red') + 'g'*(label=='green') + 'sandybrown'*(label=='total')

        kons = [self.kon[indx]]
        colors = [color]
        if load_fret: kons.append(self.kon[3]) ;colors.append('blueviolet')

        for i, kon in enumerate(kons):
            selected_sides = list(itertools.compress(['left','bottom','right'], kon))
            unselected_sides = list(itertools.compress(['left','bottom','right'], np.invert(kon)))
            [self.axes[i].spines[s].set_color(colors[i]) for s in selected_sides]
            [self.axes[i].spines[s].set_color('white') for s in unselected_sides]

        self.fig.canvas.draw_idle()

    def select_edge(self, key):
        if self.fret_edge_lock:
            ax = self.axes[0]
            label = self.radio.value_selected  # get the selected color of the radiobutton
            color = 'r'*(label=='red') + 'g'*(label=='green') + 'sandybrown'*(label=='total')
        elif not self.fret_edge_lock:
            ax = self.axes[1]
            color = 'blueviolet'  # this refers to the fret color

        side = 'left'*(key == ',')  + 'bottom'*(key == '.') + 'right'*(key == '/')

        spcolor = ax.spines[side].get_edgecolor()
        selcol, w = matplotlib.colors.to_rgba(color), matplotlib.colors.to_rgba('white')
        c = selcol*(spcolor == w) + w*(spcolor == selcol)
        ax.spines[side].set_color(c)

        self.update_kon(color, selcol, side, ax)

    def update_kon(self, sel=None, selcol=None, side=None, ax=None):
        i = ['r', 'g', 'sandybrown', 'blueviolet'].index(sel)  # These are the colors of the sides. blueviolet refers to fret
        j = ['left', 'bottom', 'right'].index(side)
        self.kon[i][j] = (ax.spines[side].get_edgecolor() == selcol)


    def check_total(self, label):
        if self.checkbtotal.get_status()[0]:
            self.ltotal.set_visible(True)
        else:
            self.ltotal.set_visible(False)
        self.checkbutton_color(label)


    def checkbutton_color(self, label):  # changes the color of the fret checkbutton. Purely for aesthetics
        if label == 'E fret':
            chbutton = self.checkbfret ; c = 'b'
        elif label == 'show total':
            chbutton = self.checkbtotal ; c = 'bisque'
        elif label == 'plot selected':
            chbutton =  self.checkbplotselected ; c = 'gold'

        if chbutton.get_status()[0]:
            chbutton.rectangles[0].set_color(c)
        elif not chbutton.get_status()[0]:
            chbutton.rectangles[0].set_color("black")
        self.fig.canvas.draw_idle()

    def save_trace(self, event=None):

        Ioff = [self.Igoff, self.Iroff]
        self.mol.plot(xlim=self.axes[0].get_xlim(),
                      ylim=self.axes[0].get_ylim(),
                      Ioff=Ioff, Imin=self.Imin,
                      Iroff=self.Iroff, Igoff=self.Igoff, save=True)

    def save_selected(self, event=None):
        selected_molecules = []
        file = self.molecules[0].file
        for mol in self.molecules:

            if mol.isSelected:
                current_file = mol.file
                if current_file == file:
                    selected_molecules.append(mol.index+1)
                else:
                    if bool(selected_molecules):
                        np.savetxt(f'{file.relativeFilePath}_selected_molecules.txt',
                                   np.array(selected_molecules, dtype=int))
                        file = current_file
                        selected_molecules = []
        if bool(selected_molecules):

            np.savetxt(f'{file.relativeFilePath}_selected_molecules.txt',
                                   np.array(selected_molecules, dtype=int))
        else:
            print('no molecules selected')

    def go_to_molecule(self, event=None):
        self.mol_indx = int(self.gotoentry.text) - 1
        self.plot_molecule()


class Draw_lines(object):
    def __init__(self, fig, iplot_radio):
        self.lines = []
        self.fig = fig
        self.radio = iplot_radio  # The InteractivePlot instance

    def onclick(self, event):
        if self.fig.canvas.manager.toolbar.mode != '':  # self.fig.canvas.manager.toolmanager.active_toggle["default"] is not None:
            return
        if event.inaxes is None:
            return
        ax = event.inaxes
        if event.button == 1:
            if ax == self.fig.get_axes()[0] or ax == self.fig.get_axes()[1]:
                sel = self.radio.value_selected*(ax == self.fig.get_axes()[0])
#                print(sel)
                sel = sel + "E"*(ax == self.fig.get_axes()[1])
                l = ax.axvline(x=event.xdata, zorder=0, lw=0.65, label="man "+sel)
                self.lines.append(l)

        if event.button == 3 and self.lines != []:
            self.lines.pop().remove()
        self.fig.canvas.draw_idle()

    def select_starttrace(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        sel = self.radio.value_selected
        l = ax.axvline(0, zorder=0, lw=0.65, c='yellow', label="man "+sel)
        self.lines.append(l)
        self.fig.canvas.draw_idle()

    def select_endtrace(self, event,endtime):
        if event.inaxes is None:
            return
        ax = event.inaxes
        sel = self.radio.value_selected
        l = ax.axvline(endtime, zorder=0, lw=0.65, c='yellow', label="man "+sel)
        self.lines.append(l)
        self.fig.canvas.draw_idle()

    def clear_all(self, event):
        while self.lines:
            self.lines.pop().remove()
        self.fig.canvas.draw_idle()

def find_nearest(array, value):
    # array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


if __name__ == '__main__':

    # Just as a working example of how the interactive plot whould be called. Here an example dataset is included inside the traces folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = Path(__file__).parents[2]

    # mainPath = Path(str(p) + '/traces')
    mainPath = os.path.join(p, WindowsPath('traces'))  # This path cannot be found strangely
    mainPath = r'C:\Users\iason\Desktop\traceanalysis\trace_analysis\traces\test_data\DNA04'
    exp = Experiment(mainPath)
    file = exp.files[0]
    molecules = file.molecules
    i = InteractivePlot(molecules)
    i.plot_initialize()
    i.plot_molecule()
    plt.show()


#self.fig.canvas.manager.toolmanager.add_tool('Next', NextTool)
#self.fig.canvas.manager.toolbar.add_tool('Next', 'foo')
#class NextTool(ToolBase, InteractivePlot):
#    '''Go to next molecule'''
#    default_keymap = 'enter, right'
#    description = 'Next Molecule 1'
#
#    def trigger(self, *args, **kwargs):
#        pass
#        InteractivePlot.__init__(InteractivePlot, self.file)
#        print(self.mol_indx
#              )
#        InteractivePlot.plot_setup(InteractivePlot)
#        print(InteractivePlot.mol)