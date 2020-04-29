# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:44:52 2018

@author: ivoseverins
"""
# from matplotlib import use
# use('wx', warn=False)

# If you get the error wxApp must be created first, restart kernel

#!/usr/bin/env python
import time
import pandas as pd
import wx #cross-platform GUI API
import wx.dataview
import wx.lib.agw.hypertreelist as HTL
from trace_analysis import Experiment, File

#Use the following lines on Mac
from sys import platform
if platform == "darwin":
    from matplotlib import use
    use('WXAgg')

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from trace_analysis.analysis import interactiveAnalysis, distributionsPanel

from trace_analysis.analysis.stepsDataAnalysis import analyze_steps
from trace_analysis.analysis import dwelltimeAnalysis

class MainFrame(wx.Frame):
    def __init__(self, parent, title):

        wx.Frame.__init__(self, parent, title='Trace Analysis', size=(390,700))


        # Status bar
        self.CreateStatusBar()

        # File menu in Menu bar
        fileMenu = wx.Menu()
        fileMenuOpen = fileMenu.Append(wx.ID_OPEN, "&Open", "Open directory")
        fileMenuFindNewFiles = fileMenu.Append(wx.ID_ANY, "&Find new files")
        fileMenuAbout = fileMenu.Append(wx.ID_ABOUT, "&About")
        fileMenuExit = fileMenu.Append(wx.ID_EXIT, "&Exit")

        self.Bind(wx.EVT_MENU, self.OnOpen, fileMenuOpen)
        self.Bind(wx.EVT_MENU, self.OnFindNewFiles, fileMenuFindNewFiles)
        self.Bind(wx.EVT_MENU, self.OnAbout, fileMenuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, fileMenuExit)

        # View menu in Menu bar
        viewMenu = wx.Menu()
        self.viewMenuShowMovie = viewMenu.AppendCheckItem(wx.ID_ANY, "&Show movie", "Show movie")
        self.viewMenuShowHistogram = viewMenu.AppendCheckItem(wx.ID_ANY,"&Show histogram", "Show histogram")

        self.Bind(wx.EVT_MENU, self.OnShowMovie, self.viewMenuShowMovie)
        self.Bind(wx.EVT_MENU, self.OnShowHistogram, self.viewMenuShowHistogram)

        # Data menu in Menu bar
        dataMenu = wx.Menu()
        self.dataMenuFindMolecules = dataMenu.Append(wx.ID_ANY, "&Find molecules", "Find molecules")
        self.dataMenuExtractTraces = dataMenu.Append(wx.ID_ANY, "&Extract traces", "Extract traces")

        self.Bind(wx.EVT_MENU, self.OnFindMolecules, self.dataMenuFindMolecules)
        self.Bind(wx.EVT_MENU, self.OnExtractTraces, self.dataMenuExtractTraces)

        # Analysis menu in Menu bar
        analysisMenu = wx.Menu()
        self.analysisMenuInteractiveAnalysis = analysisMenu.Append(wx.ID_ANY,
                        "&Interactive Analysis", "Interactive Analysis")
        self.analysisMenuAnalyzeSteps = analysisMenu.Append(wx.ID_ANY,
                            "&Analyze Steps Data", "Analyze Steps Data")
        self.analysisMenuPlotDistributions = analysisMenu.Append(wx.ID_ANY,
                                "&Dwelltime Analysis", "Dwelltime Analysis")

        self.Bind(wx.EVT_MENU, self.OnInteractiveSelection, self.analysisMenuInteractiveAnalysis)
        self.Bind(wx.EVT_MENU, self.OnAnalyzeStepsSelection, self.analysisMenuAnalyzeSteps)
        self.Bind(wx.EVT_MENU, self.OnPlotDistributionsSelection, self.analysisMenuPlotDistributions)

        # Menu bar
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu,"&File")
        menuBar.Append(viewMenu,"&View")
        menuBar.Append(dataMenu,"&Data")
        menuBar.Append(analysisMenu,"&Analysis")
        self.SetMenuBar(menuBar)


        # HyperTreeList
        self.tree = HyperTreeListPlus(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(200,300),
                                      0,
                                      wx.TR_HIDE_ROOT | HTL.TR_MULTIPLE | HTL.TR_EXTENDED |
                                      HTL.TR_HAS_BUTTONS | HTL.TR_LINES_AT_ROOT)

        self.movie = MoviePanel(parent=self)

        self.histogram = HistogramPanel(parent=self)
        self.interactive = InteractiveAnalysisPanel(self)
        self.distributions = DwelltimeAnalysisPanel(self)

        self.Show(True)


    # File menu event handlers
    def OnOpen(self,event):
        self.experimentPath = ''
        dlg = wx.DirDialog(self, "Choose a directory", self.experimentPath)
        if dlg.ShowModal() == wx.ID_OK:
            experimentPath = dlg.GetPath()
            self.experiment = Experiment(experimentPath)

            self.tree.AddExperiment(self.experiment)

            self.experiment.histogram(self.histogram.panel.axis, fileSelection = True)

            print(self.experiment.files[0].movie)

#
#            self.tree.AppendColumn('Files')
#            self.experimentRoot = self.tree.AppendItem(self.tree.GetRootItem(),exp.name)
#
#            for file in exp.files:
#                self.tree.AppendItem(self.experimentRoot, file.name)
#
#            self.tree.Expand(self.experimentRoot)

    def OnFindNewFiles(self, event):
        originalFiles = [file for file in self.experiment.files]
        self.experiment.addAllFilesInMainPath()
        newFiles = [file for file in self.experiment.files if file not in originalFiles]

        for file in newFiles:
            self.tree.AddFile(file, self.experiment.item)

        self.tree.insertDataIntoColumns()

    def OnAbout(self,event):
        dlg = wx.MessageDialog(self, 'Software for trace analysis', 'About Trace Analysis', wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnExit(self,event):
        self.Close(True) # Close program

    # View menu event handlers
    def OnShowMovie(self,event):
        if event.IsChecked():
            self.movie.Show()
            self.movie.ShowMovie()
        elif ~event.IsChecked(): self.movie.Hide()

    def OnShowHistogram(self,event):
        if event.IsChecked(): self.histogram.Show()
        elif ~event.IsChecked(): self.histogram.Hide()

    # Data menu event handlers
    def OnFindMolecules(self, event):
        for file in self.experiment.selectedFiles:
            file.find_coordinates() # The channel should be an option somewhere
        self.tree.insertDataIntoColumns()

    def OnExtractTraces(self, event):
        for file in self.experiment.selectedFiles:
            file.extract_traces()

    # Select menu event handlers
    def OnInteractiveSelection(self, event):
        if self.experiment.selectedFiles:
            molecules = [mol for file in self.experiment.selectedFiles
                         for mol in file.molecules]
    #        print(f'{file.name} dataset selected')
            self.interactive.start(molecules)
        else:
            print('No file selected')
            dlg = wx.MessageDialog(self, 'No file selected', 'Interactive analysis', wx.OK)
            dlg.ShowModal()
            dlg.Destroy()

    def OnAnalyzeStepsSelection(self, event):
        print('Analyzing Steps...')
        start = time.perf_counter()
        for file in self.experiment.selectedFiles:
            analyze_steps(file)
            print(f'Analyzing Steps for {file.name}')
        finish = time.perf_counter()
        msg = f'Steps Data analyzed for selected files. \
                \Time elapsed: {finish - start:.2f} sec.'
        dlg = wx.MessageDialog(self, msg, 'Analyze Steps Data', wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnPlotDistributionsSelection(self,event):
        if event.IsChecked(): self.distributions.Show()
        elif ~event.IsChecked(): self.distributions.Hide()


class MoviePanel(wx.Frame):
    def __init__(self, title='Movie', parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.panel = PlotPanel(self)
        self.parent = parent
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self._file = None

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        self._file = file
        self.ShowMovie()

    def ShowMovie(self, show_coordinates = True):
        if self.IsShown():
            self.panel.axis.clear()
            if self._file is not None:
                self._file.show_image(figure = self.panel.figure)

            if show_coordinates:
                if self._file.is_mapping_file:
                    self._file.mapping.show_mapping_transformation(figure=self.panel.figure)
                else:
                    self._file.show_coordinates(figure=self.panel.figure)
            self.panel.canvas.draw()
            self.panel.canvas.Refresh()

    def OnClose(self,event):
        self.parent.viewMenuShowMovie.Check(False)
        self.Hide()

class HistogramPanel(wx.Frame):
    def __init__(self, title='Histogram', parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.panel = PlotPanel(self)
        self.parent = parent
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def PlotHistogram(self, fileSelection=True, moleculeSelection = False):
        if self.IsShown():
            self.panel.axis.clear()
            self.parent.experiment.histogram(self.panel.axis, fileSelection=fileSelection)
            self.panel.canvas.draw()
            self.panel.canvas.Refresh()

    def OnClose(self,event):
        self.parent.viewMenuShowHistogram.Check(False)
        self.Hide()

class InteractiveAnalysisPanel(wx.Frame):
    def __init__(self, parent=None, title='Interactive Analysis'):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.parent = parent
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.moleculesToLoopThrough = None
        self.iPlot = None

    @property
    def currentMolecule(self):
        if self.iPlot is not None:
            return self.iPlot.mol
        else:
            print('InteractiveAnalysis not initialized')

    @property
    def currentFile(self):
        if self.iPlot is not None:
            return self.iPlot.mol
        else:
            print('InteractiveAnalysis not initialized')


    def start(self, molecules):
        self.moleculesToLoopThrough = molecules
        self.iPlot = interactiveAnalysis.InteractivePlot(molecules)
        self.iPlot.plot_initialize()
        self.iPlot.plot_molecule()

    def OnClose(self,event):
        self.parent.analysisMenuInteractiveAnalysis.Check(False)
        self.Hide()


class DwelltimeAnalysisPanel(distributionsPanel.Panel):
    def __init__(self, parent=None, title='Dwelltime Analysis'):
        # initialize parent class and build the GUI
        super().__init__(parent, title=title)
        self.parent = parent
        self.parent.Show()

    def combine_dwells_data(self):
        '''

        Returns
        -------
        dwells_data : pd.DataFrame4
            The concatenated pd.DataFrame from all the [..]_dwells_data.xlsx
            files for the selected files

        '''
        # Check if the analyze separately radiobutton is checked
        dwells_data = []
        # print(self.parent.experiment.selectedFiles)
        for file in self.parent.experiment.selectedFiles:
            filename = str(file.relativePath)+'/' + file.name +'_dwells_data.xlsx'
            d = pd.read_excel(filename, index_col=[0,1], dtype={'kon': str})
            dwells_data.append(d)

        dwells_data = pd.concat(dwells_data)#, ignore_index=True)

        return dwells_data

    # Override parent methods
    def OnPlotPress(self, event, dist=None):
        self.DistConfigSave()
        print('Plotting from GUI')
        self.save_enable()  # enable the save controls, parent method
        if dist is None:
            dist = self.comboDist.GetValue()
        else:
            pass
        # if the analyze separately radiobutton is selected:
        if self.radioDataSeparate.GetValue():
            self.figures = []
            for file in self.parent.experiment.selectedFiles:
                name = file.name
                filename = str(file.relativePath)+'/' + file.name \
                                +'_dwells_data.xlsx'
                dwells = pd.read_excel(filename, index_col=[0,1],
                                       dtype={'kon': str})
                figs = dwelltimeAnalysis.analyze(dwells, name, dist,
                                                 self.configuration[dist])
                self.figures.append(figs)


        # if the combine selected data radiobutton is selected:
        elif self.radioDataCombine.GetValue():
            name = self.entryDataName.GetValue()
            dwells = self.combine_dwells_data()
            self.figures = dwelltimeAnalysis.analyze(dwells, name, dist,
                                              self.configuration[dist])

class PlotPanel(wx.Panel):
    def __init__(self, parent, id=-1, dpi=None, **kwargs):
        wx.Panel.__init__(self, parent, id=id, size=(500,500), **kwargs)
        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2, 2))
        self.axis = self.figure.gca()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)


class HyperTreeListPlus(HTL.HyperTreeList):
    def __init__(self, arg, *args, **kwargs):
        super().__init__(arg, *args, **kwargs) # super(HyperTreeListPlus, self).__init__(arg, *args, **kwargs)

        self.AddColumn('Files', width=150)
        self.AddColumn('Molecules', width=75)
        self.AddColumn('Selected', width=75)
        self.AddColumn('Analyzed', width=75)

        self.root = self.AddRoot('root')
        self.FileItems = []

        #self.Bind(HTL.EVT_TREE_ITEM_ACTIVATED, self.OnItemActivated)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnItemActivated)
        self.Bind(HTL.EVT_TREE_ITEM_CHECKED, self.OnItemChecked)

        self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.OnItemRightClick)


#    def AppendItem(self, arg, *args, **kwargs):
#        #print(arg)
#        #print(*kwargs)
#        print(self)
#        return super().AppendItem(arg, *args, **kwargs)

    def AddExperiment(self, experiment):
        experimentItemNames = [item.GetText() for item in self.root.GetChildren()]
        if experiment.name not in experimentItemNames:
            experimentItem = self.AppendItem(self.root, experiment.name, ct_type=1, data=experiment)
            experiment.item = experimentItem

        print(experiment.name)

        for file in experiment.files:
            self.AddFile(file, experimentItem)

            #self.tree.SetItemText(item, 'test', 1)

        self.insertDataIntoColumns()
        self.Expand(experimentItem)

    def AddFile(self, file, experimentItem):

        folders = file.relativePath.parts

        nodeItemNames = [item.GetText() for item in experimentItem.GetChildren() if item.GetData() == None]

        parentItem = experimentItem
        for folder in folders:

            # Get the folderItems and folder names for the current folderItem
            nodeItems = [item for item in parentItem.GetChildren() if item.GetData() == None]
            nodeItemNames = [item.GetText() for item in nodeItems]

            if folder not in nodeItemNames:
                # Add new item for the folder and set parentItem to this item
                parentItem = self.AppendItem(parentItem, folder, ct_type = 1, data = None)
            else:
                # Set parent item to the found folderItem
                parentItem = nodeItems[nodeItemNames.index(folder)]

        item = self.AppendItem(parentItem, file.name, ct_type = 1, data = file)
        self.FileItems.append(item)

        # self.insertDataIntoColumns(item)

        return item

    def insertDataIntoColumns(self):
        for item in self.FileItems:
            itemData = item.GetData()
            if type(itemData) is File:
                self.SetItemText(item, str(len(itemData.molecules)), 1) # Should be in a different method
                self.SetItemText(item, str(len(itemData.selectedMolecules)), 2)
                N_analyzed_molecules = len([mol for mol in itemData.molecules if mol.steps is not None])
                self.SetItemText(item, str(N_analyzed_molecules), 3)

    def OnItemActivated(self, event):
        item = event.GetItem()
        itemData = item.GetData()
        if type(itemData) == File:
            self.Parent.movie.file = itemData

    def OnItemChecked(self, event):
        item = event.GetItem()
        #newItemCheckedState = bool(self.tree.GetCheckedState(item))
        newCheckedState = bool(item.IsChecked())
        self.CheckItem3(item, checked = newCheckedState)

        # itemData = item.GetData()
        # if type(itemData) == File:
        #     # file = self.GetItemPyData(item)
        #     file = itemData
        #     file.isSelected = newCheckedState

        #for child in item.GetChildren():



        #file = self.tree.GetItemData(item)

        #file = self.tree.GetItemData(item)

        # self.histogram.axis.clear()
        # self.experiment.histogram(self.histogram.axis, fileSelection = True)
        # self.histogram.canvas.draw()
        # self.histogram.canvas.Refresh()

        #print(self.h.IsShown())
#        if self.histogram.IsShown():
#            self.histogram.panel.axis.clear()
#            self.experiment.histogram(self.histogram.panel.axis, fileSelection=True)
#            self.histogram.panel.canvas.draw()
#            self.histogram.panel.canvas.Refresh()
        self.Parent.histogram.PlotHistogram()

    def OnItemRightClick(self, event):
        print('Right click')
        item = event.GetItem()
        itemData = item.GetData()

        if type(itemData) == File:
            file = itemData
            popupMenu = wx.Menu()
            popupMenuPerformMapping = popupMenu.Append(wx.ID_ANY, "&Perform mapping", "Perform mapping")
            self.Bind(wx.EVT_MENU, lambda selectEvent: self.OnPerformMapping(selectEvent, item, file),
                      popupMenuPerformMapping)

            if file.mapping is not None:
                popupMenuApplyMappingToOtherFiles = popupMenu.Append(wx.ID_ANY, "&Apply mapping to other files", "Apply mapping to other files")
                self.Bind(wx.EVT_MENU, lambda selectEvent: self.OnApplyMappingToOtherFiles(selectEvent, item, file),
                          popupMenuApplyMappingToOtherFiles)

            if file.coordinates is not None:
                popupMenuCopyCoordinatesToSelectedFiles = popupMenu.Append(wx.ID_ANY, "&Copy coordinates to selected files", "Copy coordinates to selected files")
                self.Bind(wx.EVT_MENU, lambda selectEvent: self.OnCopyCoordinatesToSelectedFiles(selectEvent, item, file),
                          popupMenuCopyCoordinatesToSelectedFiles)

            self.PopupMenu(popupMenu, event.GetPoint())

    def OnPerformMapping(self, event, item, file):
        file.perform_mapping()

    def OnApplyMappingToOtherFiles(self, event, item, file):
        file.use_mapping_for_all_files()
        standardColour = self.GetItemBackgroundColour(self.GetRootItem().GetChildren()[0])
        for item in self.FileItems: self.SetItemBackgroundColour(item, standardColour)
        self.SetItemBackgroundColour(item, wx.YELLOW) #wx.Colour(160,160,160))

    def OnCopyCoordinatesToSelectedFiles(self, event, item, file):
        # Maybe we should somehow indicate which files have coordinates from other files. [IS 31-01-2020]
        file.copy_coordinates_to_selected_files()
        self.insertDataIntoColumns()

    def CheckItem3(self, item, checked = True):
        self.CheckItem2(item, checked=checked, torefresh=True)
        itemData = item.GetData()
        if type(itemData) == File:
            #file = self.GetItemPyData(item)
            file = itemData
            file.isSelected = checked
        else:
            for child in item.GetChildren():
                self.CheckItem3(child, checked = checked)


app = wx.App(False)
frame = MainFrame(None, "MainFrame")
app.MainLoop()

print('End')

del app


"""

class MyTree(wx.TreeCtrl):
    def __init__(self, parent, id, pos, size, style):
        wx.TreeCtrl.__init__(self, parent, id, pos, size, style)

class TreePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.tree = MyTree(self, wx.ID_ANY, wx.DefaultPosition, (300,200), wx.TR_HAS_BUTTONS)

        self.root = self.tree.AddRoot('Something goes here')
        self.tree.SetPyData(self.root, ('key', 'value'))
        self.tree.AppendItem(self.root, 'Operating Systems')
        self.tree.Expand(self.root)

"""


    # TreeListCtrl event handlers
#    def Test(self, event):
#        item = event.GetItem()
#        #newItemCheckedState = bool(self.tree.GetCheckedState(item))
#        newItemCheckedState = bool(item.IsChecked())
#        #file = self.tree.GetItemData(item)
#        file = self.tree.GetItemPyData(item)
#        file.isSelected = newItemCheckedState


        # self.histogram.axis.clear()
        # self.experiment.histogram(self.histogram.axis, fileSelection = True)
        # self.histogram.canvas.draw()
        # self.histogram.canvas.Refresh()

        #print(self.h.IsShown())
#        if self.histogram.IsShown():
#            self.histogram.panel.axis.clear()
#            self.experiment.histogram(self.histogram.panel.axis, fileSelection=True)
#            self.histogram.panel.canvas.draw()
#            self.histogram.panel.canvas.Refresh()
        # self.histogram.PlotHistogram()

#class Plot(wx.Panel):
#    def __init__(self, parent, id=-1, dpi=None, **kwargs):
#        wx.Panel.__init__(self, parent, id=id, **kwargs)
#        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2, 2))
#        #self.axis = self.figure.gca()
#        self.canvas = FigureCanvas(self, -1, self.figure)
#        self.toolbar = NavigationToolbar(self.canvas)
#        self.toolbar.Realize()
#
#        sizer = wx.BoxSizer(wx.VERTICAL)
#        sizer.Add(self.canvas, 1, wx.EXPAND)
#        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
#        self.SetSizer(sizer)


#class PlotNotebook(wx.Panel):
#    def __init__(self, parent, id=-1, size = (500,500)):
#        wx.Panel.__init__(self, parent, id=id, size=size)
#        self.nb = aui.AuiNotebook(self)
#        sizer = wx.BoxSizer()
#        sizer.Add(self.nb, 1, wx.EXPAND)
#        self.SetSizer(sizer)
#
#    def add(self, name="plot"):
#        page = Plot(self.nb)
#        self.nb.AddPage(page, name)
#        return page.figure


#class TracePanel(wx.Frame):
#    def __init__(self, title='Trace', parent=None):
#        wx.Frame.__init__(self, parent=parent, title=title)
#        self.panel = PlotPanel(self)
#        self.parent = parent
#        self.Bind(wx.EVT_CLOSE, self.OnClose)
#
#        self.moleculesToLoopThrough = None
#
#        self.currentMolecule = None
#
#    def Show(self):
#        self.PlotTrace(self.currentMolecule)
#        super().Show()
#
#    def OnClose(self,event):
#        self.parent.viewMenuShowTrace.Check(False)
#        self.Hide()
#
#    def PlotTrace(self, molecule):
#        self.currentMolecule = molecule
#        if self.IsShown():
#            self.panel.axis.clear()
#            molecule.plot(figure = self.panel.figure)
#            self.panel.canvas.draw()
#            self.panel.canvas.Refresh()
#            print(str(molecule.index))
#
#    def LoopThroughMolecules(self, molecules):
#        self.moleculesToLoopThrough = molecules
#        self.panel.canvas.Bind(wx.EVT_KEY_DOWN, self.GoToNextMolecule)
#        self.Show()
#        self.PlotTrace(molecules[0])
#        print('LoopThrough')


        #self.tree = HTL.HyperTreeList(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(200,300),
        #                              HTL.TR_MULTIPLE | HTL.TR_EXTENDED)



        # TreeListCtrl
        #self.tree = wx.dataview.TreeListCtrl(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(200,300),
        #                                     wx.dataview.TL_CHECKBOX | wx.dataview.TL_MULTIPLE)
        #self.Bind(wx.dataview.EVT_TREELIST_ITEM_CHECKED, self.Test, self.tree)
        #self.Bind(wx.EVT_TREE_SEL_CHANGED, self.Test, self.tree)
        ##self.tree.Bind(wx.EVT_LEFT_DOWN, self.Test)
        #panel = TreePanel(self)


        #test = wx.Button(self, -1, 'Large button')
        #test = wx.Button(self, -1, 'Large button')
        #wx.Button(self.panel1, -1, 'Small button')
        #box = wx.BoxSizer(wx.HORIZONTAL)
        #box.Add(self.tree, 0,wx.EXPAND,0)
        #box.Add(self.plotter, 0,0,0)
        #box.Add(self.histogram,1,wx.EXPAND,0)

        #box.Add(wx.Button(self, -1, 'Small button'), 0, 0, 0)
        #box.Add(wx.Button(self, -1, 'Large button'), 0, 0, 0)
        #box.Add(self.tree, 1,0,0)
        #box.Add(self.panel2, 1,0,0)
        #self.SetSizerAndFit(box)
