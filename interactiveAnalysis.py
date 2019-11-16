# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:12:58 2019

@author: iason
"""

import numpy as np
import sys

import traceAnalysisCode as analysis
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.widgets
import seaborn as sns
from pathlib import Path, PureWindowsPath
import functools
from PIL import Image
#plt.rcParams['toolbar'] = 'toolmanager'
#from matplotlib.backend_tools import ToolBase
#mainPath = r'D:\ivoseverins\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\HJ A'


class InteractivePlot(object):
    def __init__(self, file, import_excel=True):
        plt.ioff()
        self.file = file
        self.mol_indx = 0  #   From which molecule to start the analysis
        self.exp_time = self.file.exposure_time
        self.time = np.arange(0, len(self.file.molecules[0].I(0)))*self.exp_time  # the array is multiplied with exposure time in the end to ensure equal dimensions
        #  See if there are saved analyzed molecules
        if import_excel:
            self.file.importExcel(filename=self.file.name+'_steps_data.xlsx')

    def plot_initialize(self):

        sns.set(style="dark")
        sns.set_color_codes()
        #plt.style.use('dark_background')

        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 7))
        self.fig.canvas.set_window_title(f'Dataset: {self.file.name}')
        self.axes[0].axis('off')
        self.axes[1].set_ylim((-50, 500))  # Det default intensity limits
        self.axes[1].set_ylabel("intensity (a.u)\n")
        self.axes[1].set_xlim((-2, self.time[-1] + 2))
        self.axes[1].set_xticklabels([])
        self.axes[2].set_ylim((-0.1, 1.1))  # Det default fret limits
        self.axes[2].set_xlabel("time (s)")
        self.axes[2].set_ylabel("FRET\n")
        self.axes[2].set_xlim((-2, self.time[-1] + 2))  # Set default x limits dependent on measurement time

        #plt.subplots_adjust(top=0.75)
        plt.subplots_adjust(bottom=0.23)

        # Create the axes for the widgets
        #self.axROI = plt.axes([0.6, 0.8, 0.2, 0.15])
        self.rax = plt.axes([0.91, 0.72, 0.08, 0.15])
        self.axtotal = plt.axes([0.91, 0.66, 0.08, 0.08])

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

        self.axnextb = plt.axes([0.17, 0.90, 0.065, 0.062])  # Buttons to cycle through molecules
        self.axprevb = plt.axes([0.083, 0.90, 0.08, 0.062])
        [ax.set_frame_on(False) for ax in self.fig.get_axes()[3:]]

        #  Radiobutton to select red or green
        self.radio = matplotlib.widgets.RadioButtons(self.rax, ("red", "green"))
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

        #  A checkbutton for whether to display the total intensity
        self.checkbtotal = matplotlib.widgets.CheckButtons(self.axtotal, ["Total"],
                                                           actives=[False])

        #  A checkbutton for fret autothreshold dwell-time calculation
        self.checkbfret = matplotlib.widgets.CheckButtons(self.axcheckfret, ["E fret"],
                                                          actives=[False])


        for chbutton in [self.checkbtotal, self.checkbfret]:
            chbutton.rectangles[0].set_color("black")
            chbutton.rectangles[0].set_height( 0.2)
            [line.remove() for line in chbutton.lines[0]]

        #  Entryboxes for offset corrections
        corrdict = {'initial': str(0), 'color':'k', 'hovercolor': "k", 'label_pad':.2}
        corrlabels = [r'$I_{R_{off}}$', r'$I_{G_{off}}$', r'$I_{min}$']
        corraxes = [self.axcorred, self.axcorgreen, self.axcorrfretI]
        self.correntries = [matplotlib.widgets.TextBox(ax, label, **corrdict)
                            for ax, label in zip(corraxes, corrlabels)]

        #  Sliders for assigning the threshold
        self.thrsliders = []
        self.thrsliders.append(matplotlib.widgets.Slider(self.axthrsliders[0],
                                                         label=r"$I_R$", valmin=0,
                                                         valmax=500, valinit=100,
                                                         valfmt="%i", color="r"))
        self.thrsliders.append(matplotlib.widgets.Slider(self.axthrsliders[1],
                                                         label=r"$E$", valmin=0,
                                                         valfmt="%.2f", valinit=0.5,
                                                         color="b", valmax=1.0))
        [slider.vline.remove() for slider in self.thrsliders]  # remove the default vertical lines showing the initial value

        self.connect_events_to_canvas()
        self.fig.show()

    def connect_events_to_canvas(self):
        self.fig.canvas.mpl_connect('button_press_event', self.draw.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.key_bind)
        self.fig.canvas.mpl_connect('axes_leave_event', functools.partial(self.change_axis, 'leave'))
        self.fig.canvas.mpl_connect('axes_enter_event', functools.partial(self.change_axis, 'enter'))
        [entry.on_submit(lambda _: self.plot_molecule()) for entry in self.correntries]
        [slider.on_changed(functools.partial(self.change_slider, slider)) for slider in self.thrsliders]
        self.checkbtotal.on_clicked(self.check_total)
        self.checkbfret.on_clicked(self.checkbutton_color)

    def plot_molecule(self, draw_plot=True):
        # clear the appropriate lines from axes first
        [ax.lines.clear() for ax in self.fig.get_axes()[:3]]
        # find the current molecule instance
        self.mol = self.file.molecules[self.mol_indx]

        # Check if molecule is selected
        if self.mol.isSelected:
            self.select_molecule(toggle=False)
        else:
            self.select_molecule(toggle=False, deselect=True)
        # load saved steps if available
        self.load_from_Molecule()

        # load kon if existing or assign a False 3x3 boolean
        self.prev_mol = self.file.molecules[self.mol_indx - 1]
        if all(kon is None for kon in [self.mol.kon_boolean, self.prev_mol.kon_boolean]):
            self.kon = np.zeros((3, 3), dtype=bool)
        elif self.mol.kon_boolean is None:
            self.kon = np.copy(self.prev_mol.kon_boolean)  # if no kon is defined for current molecule
        else:
            self.kon = self.mol.kon_boolean

        # update the edge color from self.kon:
        self.load_edges(load_fret=True)

        self.Iroff, self.Igoff, self.Imin = [float(c.text) for c in self.correntries]

        self.red = self.mol.I(1, Ioff=self.Iroff)
        self.green = self.mol.I(0, Ioff=self.Igoff)

        self.fret = self.mol.E(Imin=self.Imin, Iroff=self.Iroff, Igoff=self.Igoff)

        if not draw_plot:
            return

        #  if draw_plot is true
        self.axes[1].plot(self.time, self.green, "g", lw=.75)
        self.axes[1].plot(self.time, self.red, "r", lw=.75)
        self.total = self.red + self.green
        self.ltotal = self.axes[1].plot(self.time, self.total, "bisque", lw=.65,
                               zorder=0, visible=self.checkbtotal.get_status()[0])[0]

        self.axes[2].plot(self.time, self.fret, "b", lw=.75)
   
        if self.file.avgImage is not None:
            self.axes[0].imshow(self.plotROI())

        # vertical lines to indicate the threshold in the two axes
        self.slidel = [ax.axhline(0, lw=1, ls=":", zorder=3, visible=False) for ax in self.axes[1:]]
        #  Creat cursor particular to the molelcule and connect it to mouse movement event
        self.cursors = []
        cursor_kws = {'useblit': True, 'color': 'black', 'ls': "--", 'lw': 1, 'alpha': 0.5}
        self.cursors.append(matplotlib.widgets.Cursor(self.axes[1], **cursor_kws))
        self.cursors.append(matplotlib.widgets.Cursor(self.axes[2], **cursor_kws))

        self.fig.canvas.draw_idle()
        
    def plotROI(self):
        img = self.file.avgImage
        bnd = 5
        totwidth = 4*bnd+2
        totheight = 2*bnd+1
        coor = self.file.molecules[self.mol_indx].coordinates
        ROI1 = img.crop((coor[0]-bnd-1, coor[1]-bnd-1, coor[0]+bnd, coor[1]+bnd))
        ROI2 = img.crop((coor[2]-bnd-1, coor[3]-bnd-1, coor[2]+bnd, coor[3]+bnd))
        ROI_mrg = Image.new("P", (totwidth, totheight))
        ROI_mrg.paste(ROI1, (0, 0))
        ROI_mrg.paste(ROI2, (2*bnd+1, 0))
        pixels = ROI_mrg.load()
        
        for x in range(2*bnd+1):
            for y in range(2*bnd+1):
                if (x-bnd)**2 + (y-bnd)**2 == (bnd)**2:
                    pixels[x, y] = int(np.max(ROI_mrg.getextrema()))
                    pixels[x+2*bnd+1, y] = int(np.max(ROI_mrg.getextrema()))
        return ROI_mrg

    def change_axis(self, event_type, event):
        ax = event.inaxes
        if event_type == 'enter':
            if ax == self.axes[2]:
                self.fret_edge_lock = False
            elif ax in self.axthrsliders:
                indx = int(ax == self.axthrsliders[1])  # gives 0 if ax is upper (I) plot, 1 if ax is lower (E) plot
                self.slidel[indx].set_ydata(self.thrsliders[indx].val)
                self.slidel[indx].set_visible(True)
                self.fig.canvas.draw()
        elif event_type in ['leave', 'slider_change']:
            if ax == self.axes[2]:
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
        elif k in ['left', 'right']: self.save_molecule(event, move=True)
        elif k == 'z': self.auto_reject(event)
        elif k == 'c': self.draw.clear_all(event)
        elif k in [',', '.', '/']: self.select_edge(k)
        elif k == ' ': self.select_molecule(toggle=True)
        elif k == 'r': self.radio_manage('red')
        elif k == 'g': self.radio_manage('green')
        elif k == 'e':
            self.checkbfret.set_active(0)
            self.checkbutton_color('E fret')
        elif k == 'i':
            self.checkbtotal.set_active(0)
            self.check_total('Total')

        elif k == 't': self.throw_away(event)
        elif k == 'l': self.conclude_analysis()
        elif k == '[': self.draw.select_starttrace(event)
        elif k == ']': self.draw.select_endtrace(event, self.time[-1])

        self.fig.canvas.draw_idle()


    def load_from_Molecule(self):
        if self.mol.steps is None:
            return
        else:
            s = self.mol.steps
            [self.axes[1].axvline(f, zorder=0, lw=0.65, label="saved r")
             for f in s.time[s.trace == 'red'].values]
            [self.axes[1].axvline(f, zorder=0, lw=0.65, label="saved g")
             for f in s.time[s.trace == 'green'].values]
            [self.axes[2].axvline(f, zorder=0, lw=0.65, label="saved E")
             for f in s.time[s.trace == 'E'].values]

    def select_molecule(self, toggle=True, deselect=False):
        if toggle:
            self.mol.isSelected = not self.mol.isSelected
        elif deselect:
            self.mol.isSelected = False
        else:
            self.mol.isSelected = True

        title = f'Molecule: {self.mol.index} /{len(self.file.molecules)}'
        title += '  (S)'*(self.mol.isSelected)
        rgba = matplotlib.colors.to_rgba
        c = rgba('g')*self.mol.isSelected + rgba('b')*(not self.mol.isSelected)
        self.axes[1].set_title(title, color=c)
        self.fig.canvas.draw_idle()

    def throw_away(self, event):
        if self.mol.steps is not None:
            self.mol.steps = None
            lines = self.axes[1].get_lines() + self.axes[2].get_lines()
            [l.remove() for l in lines if l.get_label().split()[0] in ['man', 'thres', 'saved']]
            self.select_molecule(toggle=False, deselect=True)
            self.fig.canvas.draw_idle()


    def save_molecule(self, event=None, move=True, draw=True):
#       Assume acceptance of automatically found and manually selected dwell times
        lines = self.axes[1].get_lines() + self.axes[2].get_lines()
        lines = [l for l in lines if l.get_label().split()[0] in ["man", "thres"]]
        self.mol.kon_boolean = self.kon
        if lines:
            if len(lines) % 2 != 0:
                print(f'Found an odd number of steps. Molecule {self.mol.index} not added')
                return
            if self.mol.steps is None:
                self.mol.steps = pd.DataFrame(columns=['time', 'trace', 'state',
                                                       'method','thres','kon',
                                                       'Iroff', 'Igoff', 'Imin'])
            self.mol.isSelected = True  # Molecule is not automatically selected if steps are indicated
            kon = [f'{int(i)}' for i in self.mol.kon_boolean.flatten()]
            kon = ''.join(kon)

            for l in lines:
                method = l.get_label().split()[0]
                thres = "N/A" * (method == 'man') + str(self.thrsliders[0].val) * (method == 'thres')
                d = {'time': l.get_xdata()[1], 'trace': l.get_label().split()[1],
                     'state': 1, 'method': method, 'thres': thres, 'kon': kon,
                      'Iroff': self.Iroff, 'Igoff': self.Igoff, 'Imin': self.Imin}
                self.mol.steps = self.mol.steps.append(d, ignore_index=True)
            self.mol.steps.drop_duplicates(inplace=True)

#           Sort the timepoints
            self.mol.steps = self.mol.steps.sort_values(by=['time'])
            self.mol.steps.reset_index(inplace=True, drop=True)

        if move:
            if event.inaxes == self.axnextb or event.key in ['right']:
                if self.mol_indx >= len(self.file.molecules):
                    self.mol_indx = 0
                else:
                    self.mol_indx += 1

            elif event.inaxes == self.axprevb or event.key in ['left']:
                self.mol_indx -= 1

            self.plot_molecule(draw_plot=draw)

    def conclude_analysis(self, event=None, save=True, filename=None):
        # Save current molecule if it was analyzed
        self.save_molecule(move=False)

        if filename is None:
            filename = self.file.name+'_steps_data.xlsx'

        if save:
            self.file.savetoExcel(filename=filename)


    def autoThreshold_plot(self, event=None, find_all=False):
        self.auto_reject()
        #  Find the steps for the checked buttons
        sel = self.radio.value_selected
        color = self.red*bool(sel == "red") + self.green*bool(sel == "green")  # Select trace data for red  or green
        steps = self.mol.find_steps(color, threshold=self.thrsliders[0].val)
        l_props = {"lw": 0.75, "zorder": 5, "label": "thres "+sel}
        [self.axes[1].axvline(s*self.exp_time, **l_props) for s in steps["frames"]]
        if self.checkbfret.get_status()[0]:
            steps = self.mol.find_steps(self.fret, threshold=self.thrsliders[1].val)
            l_props = {"lw": 0.75, "zorder": 5, "label": "thres E"}
            [self.axes[2].axvline(s*self.exp_time, **l_props) for s in steps["frames"]]
        self.fig.canvas.draw()
        if find_all:
            for mol in self.file.molecules:
                self.autoThreshold_plot(find_all=False)
                print(f'Analyzed mol {self.mol.index} /{len(self.file.molecules)}')
                e = matplotlib.backend_bases.KeyEvent('key_press_event', self.fig.canvas, 'right')
                if mol != self.file.molecules[-1]:
                    self.save_molecule(event=e, move=True, draw=False)
                elif mol == self.file.molecules[-1]:
                    self.conclude_analysis()
                    return

    def auto_reject(self, event=None):
        for ax in self.axes:
            lines = ax.get_lines()
            [l.remove() for l in lines if l.get_label().split()[0] == 'thres']
            self.fig.canvas.draw_idle()


    def radio_manage(self, label):
        def update_slider(color, label):
            s = self.thrsliders[0]
            s.poly.set_color(color); s.label.set(text=label)

        indx = int(label == 'green')  # 1 if green, 0 if red
        self.axes[1].get_lines()[not indx].set_zorder((not indx)+2)
        self.axes[1].get_lines()[indx].set_zorder(indx)
        self.radio.circles[indx].set_color(label[0])
        self.radio.circles[not indx].set_color("black")
        update_slider(label[0], r"$I_G$"*bool(indx)+r"$I_R$"*bool(not indx))
        # Check the edge colors and set to white if not selected color
        sel = self.radio.value_selected
        selcol = matplotlib.colors.to_rgba(sel[0])
        spcol = [self.axes[1].spines[s].get_edgecolor() for s in ['left','bottom','right']]
        if selcol not in spcol:
            [self.axes[1].spines[s].set_color('black') for s in ['left','bottom','right']]

        self.load_edges()

    def load_edges(self, load_fret=False):  # loads edge color from kon array
        sel = self.radio.value_selected
        kons = [self.kon[int(sel == 'green')]] ; colors = [sel[0]]
        if load_fret: kons.append(self.kon[2]) ;colors.append('blueviolet')

        for i, kon in enumerate(kons):
            selected_sides = list(itertools.compress(['left','bottom','right'], kon))
            unselected_sides = list(itertools.compress(['left','bottom','right'], np.invert(kon)))
            [self.axes[i].spines[s].set_color(colors[i]) for s in selected_sides]
            [self.axes[i].spines[s].set_color('white') for s in unselected_sides]

        self.fig.canvas.draw_idle()

    def select_edge(self, key):
        if self.fret_edge_lock:
            ax = self.axes[1]
            sel = self.radio.value_selected[0]  # get the selected color of the radiobutton
        elif not self.fret_edge_lock:
            ax = self.axes[2]
            sel = 'blueviolet'  # this refers to the fret color

        side = 'left'*(key == ',')  + 'bottom'*(key == '.') + 'right'*(key == '/')

        spcolor = ax.spines[side].get_edgecolor()
        selcol, w = matplotlib.colors.to_rgba(sel), matplotlib.colors.to_rgba('white')
        c = selcol*(spcolor == w) + w*(spcolor == selcol)
        ax.spines[side].set_color(c)

        self.update_kon(sel, selcol, side, ax)

    def update_kon(self, sel=None, selcol=None, side=None, ax=None):
        i = ['r', 'g', 'blueviolet'].index(sel)  # These are the colors of the sides. blueviolet refers to fret
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
        elif label == 'Total':
            chbutton = self.checkbtotal ; c = 'darkgoldenrod'

        if chbutton.get_status()[0]:
            chbutton.rectangles[0].set_color(c)
        elif not chbutton.get_status()[0]:
            chbutton.rectangles[0].set_color("black")
        self.fig.canvas.draw_idle()

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
            if ax == self.fig.get_axes()[1] or ax == self.fig.get_axes()[2]:
                sel = self.radio.value_selected*(ax == self.fig.get_axes()[1])
                sel = sel + "E"*(ax == self.fig.get_axes()[2])
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

if __name__ == '__main__':
    # Just as a working example of how the interactive plot whould be called. Here an example dataset is included inside the traces folder
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        mainPath = './traces'
        mainPath = Path(mainPath)
        exp = analysis.Experiment(mainPath)
        file = exp.files[0]
        file.exposure_time = 0.3  # Here given explicitly because there is no .log file
        i = InteractivePlot(file)
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