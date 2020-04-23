# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 01:02:04 2019

@author: iason
"""
import wx

class Panel(wx.Frame):

    def __init__(self, parent=None, title='Plot and Fit dwelltime distributions'):
        wx.Frame.__init__(self, parent, title=title)

        self.InitUI()
        self.Centre()

    def InitUI(self):

        panel = wx.Panel(self)
        #  Top sizer which places the different boxes vertically
        topsizer = wx.BoxSizer(wx.VERTICAL)

        #  SizerFlags object for commonly used flags
        flagsExpand = wx.SizerFlags(0)
        flagsExpand.Expand().Border(wx.ALL, 5)

        flagsCenter = wx.SizerFlags(0)
        flagsCenter.Align(wx.CENTER).Border(wx.ALL, 5)

        flagsLeft = wx.SizerFlags(0)
        flagsLeft.Align(wx.LEFT).Border(wx.ALL, 5)

        # Grid sizer to hold dataset selection radiobutton and naming option
        sboxData = wx.StaticBox(panel, label="Data")
        sboxDataSizer = wx.StaticBoxSizer(sboxData, wx.VERTICAL)

        gridSizerDataSelect = wx.GridBagSizer(5, 5)
        self.radioDataSeparate = wx.RadioButton(sboxData,
                                                label='Analyze separately',
                                                style = wx.RB_GROUP)

        self.radioDataSeparate.Bind(wx.EVT_RADIOBUTTON, self.onRadioSeparatePress)

        self.radioDataCombine = wx.RadioButton(sboxData,
                                                label='Combine selected')

        self.radioDataCombine.Bind(wx.EVT_RADIOBUTTON, self.onRadioCombinePress)
        self.entryDataName = wx.TextCtrl(sboxData, size=(80,20))

        self.entryDataName.Disable()
        gridSizerDataSelect.Add(self.radioDataSeparate, pos=(0, 0), flag=wx.LEFT,
                                border=5)
        gridSizerDataSelect.Add(self.radioDataCombine, pos=(1, 0), flag=wx.LEFT,
                                border=5)
        gridSizerDataSelect.Add(wx.StaticText(sboxData, label='Dataset Name:'),
                                pos=(0, 1), flag=wx.LEFT, border=5)
        gridSizerDataSelect.Add(self.entryDataName, pos=(1, 1), flag=wx.LEFT,
                                border=5)

        sboxDataSizer.Add(gridSizerDataSelect, 1, wx.EXPAND)



        # Static box for input of distribution plot and fit options
        sboxConfig = wx.StaticBox(panel, label="Configuration")
        sboxConfigSizer = wx.StaticBoxSizer(sboxConfig, wx.VERTICAL)

        # Box to hold the distribution type choice
        sBoxSizerSelect = wx.BoxSizer(wx.HORIZONTAL)
        sBoxSizerSelect.Add(wx.StaticText(sboxConfig, label="Distribution: ",
                                          style=wx.ALIGN_LEFT), flagsCenter)



        self.comboDist = wx.ComboBox(sboxConfig, value='offtime',
                                        choices=['offtime', 'ontime'],
                                        style = wx.ALIGN_CENTRE)
        # Bind events to combobox. When the box is clicked
        self.comboDist.Bind(wx.EVT_COMBOBOX, self.DistConfigLoad)
        self.comboDist.Bind(wx.EVT_COMBOBOX_DROPDOWN, self.DistConfigSave)
        self.configuration = {key: {} for key in self.comboDist.GetItems()}
        # configparams = ['trace', 'side', 'min', 'max', 'scale', 'PlotType',
        #                 'binsize', 'FitBool', 'TmaxBool', 'model', 'Nfits']

        sBoxSizerSelect.Add(self.comboDist, flagsCenter)

        # GridBag Sizer to hold the parameter selection buttons
        gridSizer = wx.GridBagSizer(5, 5)
        # 1st column, choose the trace type(s)
        gridSizer.Add(wx.StaticText(sboxConfig, label='Trace:'),  pos=(0, 0),
                      flag=wx.LEFT, border=5)
        # 2nd column, choose the trace type(s)
        gridSizer.Add(wx.StaticText(sboxConfig, label='Side:'),  pos=(0, 1),
                      flag=wx.LEFT, border=5)

        traces = ['red', 'green', 'total', 'FRET']
        self.chbTraces = [wx.CheckBox(sboxConfig, label=l) for l in traces]
        sides = ['left', 'middle', 'right']
        self.chbSides = [wx.CheckBox(sboxConfig, label=l) for l in sides]

        for i, chb in enumerate(self.chbTraces):
            gridSizer.Add(chb, pos=(i+1,0), flag=wx.LEFT, border=5)

        for i, chb in enumerate(self.chbSides):
            gridSizer.Add(chb, pos=(i+1,1), flag=wx.LEFT, border=5)

        # Entry Boxes for min and max dwelltimes
        boxSizerMin = wx.BoxSizer(wx.HORIZONTAL)
        boxSizerMin.Add(wx.StaticText(sboxConfig, label='Min:'), flagsLeft)
        self.entryMinDwell = wx.TextCtrl(sboxConfig, value="0", size=(40,20))
        boxSizerMin.Add(self.entryMinDwell, flagsLeft)

        boxSizerMax = wx.BoxSizer(wx.HORIZONTAL)
        boxSizerMax.Add(wx.StaticText(sboxConfig, label='Max:'), flagsLeft)
        self.entryMaxDwell = wx.TextCtrl(sboxConfig, value="Max", size=(40,20))
        boxSizerMax.Add(self.entryMaxDwell, flagsLeft)

        gridSizer.Add(boxSizerMin,  pos=(5, 0), span=(1,2), flag=wx.LEFT, border=2)
        gridSizer.Add(boxSizerMax,  pos=(6, 0), span=(1,2), flag=wx.LEFT, border=2)

        # Controls for plot-related parameters input
        boxSizerPlotParams = wx.StaticBoxSizer(wx.HORIZONTAL, sboxConfig,
                                               label="Plot options")
        gridSizerPlotParams = wx.GridBagSizer(5, 5)
        # Add the scale choose combobox
        gridSizerPlotParams.Add(wx.StaticText(sboxConfig, label='Scale:'),
                                pos=(0, 0), flag=wx.LEFT, border=5)
        self.comboScale = wx.ComboBox(sboxConfig, value='Normal',
                                        choices=['Normal', 'Log', 'Log-Log'],
                                        style = wx.ALIGN_CENTRE)
        gridSizerPlotParams.Add(self.comboScale, pos=(0,1), flag=wx.LEFT,
                                 border=5)

        # Add the plot type choose combobox
        gridSizerPlotParams.Add(wx.StaticText(sboxConfig, label='Plot type:'),
                                pos=(1, 0), flag=wx.LEFT, border=5)
        self.comboPlotType = wx.ComboBox(sboxConfig, value='dots',
                                        choices=['dots', 'bars', 'line'],
                                        style = wx.ALIGN_CENTRE)
        gridSizerPlotParams.Add(self.comboPlotType, pos=(1,1), flag=wx.LEFT,
                                 border=5)

        # Add the plot binning entry box
        gridSizerPlotParams.Add(wx.StaticText(sboxConfig, label='Bin size:'),
                                pos=(0, 2), flag=wx.LEFT, border=5)
        self.entryBinSize = wx.TextCtrl(sboxConfig, value="auto", size=(45,20))
        gridSizerPlotParams.Add(self.entryBinSize, pos=(0,3), flag=wx.LEFT,
                                 border=5)
        boxSizerPlotParams.Add(gridSizerPlotParams, flagsLeft)

        # Controls for fit-related parameters input
        boxSizerFitParams = wx.StaticBoxSizer(wx.VERTICAL, sboxConfig,
                                               label="Fit options")

        gridSizerFitParams = wx.GridBagSizer(5, 5)


        self.chbFitBool = wx.CheckBox(sboxConfig, label='Fit')
        self.chbTmaxBool = wx.CheckBox(sboxConfig, label='Include > Tmax')
        self.chbBootBool = wx.CheckBox(sboxConfig, label='Bootstrap')
        self.comboFitModel = wx.ComboBox(sboxConfig, value='1Exp',
                                        choices=['1Exp', '2Exp',
                                                 '1Exp+2Exp', '3Exp'],
                                        style = wx.ALIGN_CENTRE)
        self.entryNfits = wx.TextCtrl(sboxConfig, value="10", size=(40,20))
        self.entryBoots = wx.TextCtrl(sboxConfig, value="100", size=(40,20))

        gridSizerFitParams.Add(self.chbFitBool, pos=(0,0),
                               flag=wx.LEFT, border=5)
        gridSizerFitParams.Add(self.chbTmaxBool, pos=(1,0),
                               flag=wx.LEFT, border=5)
        gridSizerFitParams.Add(self.chbBootBool, pos=(2,0),
                               flag=wx.LEFT, border=5)

        gridSizerFitParams.Add(wx.StaticText(sboxConfig, label='Model:'),
                                pos=(0, 1), flag=wx.LEFT, border=0)
        gridSizerFitParams.Add(wx.StaticText(sboxConfig, label='Best of:'),
                                pos=(1, 1), flag=wx.LEFT, border=0)
        gridSizerFitParams.Add(wx.StaticText(sboxConfig, label='Repeats:'),
                                pos=(2, 1), flag=wx.LEFT, border=0)
        gridSizerFitParams.Add(self.comboFitModel, pos=(0,3),
                               flag=wx.LEFT, border=0)
        gridSizerFitParams.Add(self.entryNfits, pos=(1,3),
                               flag=wx.LEFT, border=0)
        gridSizerFitParams.Add(self.entryBoots, pos=(2,3),
                               flag=wx.LEFT, border=0)

        boxSizerFitParams.Add(gridSizerFitParams)

        sboxConfigSizer.Add(sBoxSizerSelect, 1, wx.EXPAND)
        sboxConfigSizer.Add(gridSizer, 4, wx.EXPAND)
        sboxConfigSizer.Add(boxSizerPlotParams, 2, wx.EXPAND)
        sboxConfigSizer.Add(boxSizerFitParams, 2, wx.EXPAND)


        # Controls and button for save options
        self.gridSaveSizer = wx.GridBagSizer(5, 5)
        self.SavePlotsBool = wx.CheckBox(panel, label='Save Plots')
        self.SaveFitsBool = wx.CheckBox(panel, label='Save Fits')
        self.SaveButton = wx.Button(panel, label='Save')


        self.gridSaveSizer.Add(self.SavePlotsBool, pos=(0,0),
                               flag=wx.LEFT, border=5)
        self.gridSaveSizer.Add(self.SaveFitsBool, pos=(1,0),
                               flag=wx.LEFT, border=5)
        self.gridSaveSizer.Add(self.SaveButton, pos=(0,1), span=(2,1),
                               flag=wx.LEFT, border=5)

        for child in self.gridSaveSizer.GetChildren():
            child.GetWindow().Disable()

        # Control buttons
        boxPlotButtonsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.PlotButton = wx.Button(panel, label='Plot')
        self.PlotAllButton = wx.Button(panel, label='Plot All')
        self.PlotAllButton.Disable()
        boxPlotButtonsSizer.AddMany([(self.PlotButton, flagsCenter),
                                     (self.PlotAllButton, flagsCenter)])


        boxControlButtonsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.HelpButton = wx.Button(panel, label='Help')
        self.ClearButton = wx.Button(panel, label='Clear')
        self.ClearButton.Disable()
        self.QuitButton = wx.Button(panel, label='Quit')
        boxControlButtonsSizer.AddMany([(self.HelpButton, flagsCenter),
                                     (self.ClearButton, flagsCenter),
                                     (self.QuitButton, flagsCenter)])

        self.PlotButton.Bind(wx.EVT_BUTTON, self.OnPlotPress)
        self.PlotAllButton.Bind(wx.EVT_BUTTON, self.OnPlotAllPress)
        self.HelpButton.Bind(wx.EVT_BUTTON, self.OnHelpPress)
        self.ClearButton.Bind(wx.EVT_BUTTON, self.OnClearPress)
        self.QuitButton.Bind(wx.EVT_BUTTON, self.OnQuitPress)

        #  Add the main sizers to the top sizer
        topsizer.Add(sboxDataSizer, flagsExpand)
        topsizer.Add(sboxConfigSizer, flagsExpand)
        topsizer.Add(self.gridSaveSizer, flagsExpand)
        topsizer.Add(boxPlotButtonsSizer, flagsExpand)
        topsizer.Add(boxControlButtonsSizer, flagsExpand)

        # topsizer.SetSizeHints(self)
        panel.SetSizer(topsizer)
        topsizer.Fit(self)

        # save a zero configuration for convenience
        self.zero_configuration = self.DistConfigSave()
        for key in self.configuration.keys():
            self.configuration[key] = self.zero_configuration.copy()

        # Hold all the parameter widgets in a list for convenience
        self.param_widgets = []

    def DistConfigLoad(self, event):
        # print(self.comboDist.GetValue(), 'Loading configuration')
        dist = self.comboDist.GetValue()

        # load the trace type(s) to be plotted
        for chbox in self.chbTraces:
            chbox.SetValue(self.configuration[dist]['trace'][chbox.GetLabel()])

        for chbox in self.chbSides:
            chbox.SetValue(self.configuration[dist]['side'][chbox.GetLabel()])

        self.entryMinDwell.SetValue(self.configuration[dist]['min'])
        self.entryMaxDwell.SetValue(self.configuration[dist]['max'])

        self.comboScale.SetValue( self.configuration[dist]['scale'])
        self.comboPlotType.SetValue( self.configuration[dist]['PlotType'])
        self.entryBinSize.SetValue(self.configuration[dist]['binsize'])

        self.chbFitBool.SetValue(self.configuration[dist]['FitBool'])
        self.chbTmaxBool.SetValue(self.configuration[dist]['TmaxBool'])
        self.chbBootBool.SetValue(self.configuration[dist]['BootBool'])
        self.comboFitModel.SetValue(self.configuration[dist]['model'])
        self.entryNfits.SetValue(self.configuration[dist]['Nfits'])
        self.entryBoots.SetValue(self.configuration[dist]['BootRepeats'])

        return self.configuration[dist]


    def DistConfigSave(self, event=None):
        dist = self.comboDist.GetValue()
        # print(dist, 'Saving configuration')
        # save the trace type(s) to be plotted
        self.configuration[dist]['trace'] = {}
        for chbox in self.chbTraces:
            self.configuration[dist]['trace'][chbox.GetLabel()] = chbox.GetValue()
        # save the side(s) to be plotted
        self.configuration[dist]['side'] = {}
        for chbox in self.chbSides:
            self.configuration[dist]['side'][chbox.GetLabel()] = chbox.GetValue()

        # save the min and max values
        self.configuration[dist]['min'] = self.entryMinDwell.GetValue()
        self.configuration[dist]['max'] = self.entryMaxDwell.GetValue()

        # save the plot options
        self.configuration[dist]['scale'] = self.comboScale.GetValue()
        self.configuration[dist]['PlotType'] = self.comboPlotType.GetValue()
        self.configuration[dist]['binsize'] = self.entryBinSize.GetValue()

        # save the fit options
        self.configuration[dist]['FitBool'] = self.chbFitBool.GetValue()
        self.configuration[dist]['TmaxBool'] = self.chbTmaxBool.GetValue()
        self.configuration[dist]['BootBool'] = self.chbBootBool.GetValue()

        self.configuration[dist]['model'] = self.comboFitModel.GetValue()
        self.configuration[dist]['Nfits'] = self.entryNfits.GetValue()
        self.configuration[dist]['BootRepeats'] = self.entryBoots.GetValue()

        return self.configuration[dist]

    def onRadioCombinePress(self, event):
        self.entryDataName.Enable()

    def onRadioSeparatePress(self, event):
        self.entryDataName.Disable()

    def OnHelpPress(self, event):
        print('Help')

    def OnPlotPress(self, event):
        print('Plot')
        # self.save_enable()

    def OnPlotAllPress(self, event):
        print('Plot All')
        # self.save_enable()

    def OnClearPress(self, event):
        print('Clear')

    def OnQuitPress(self, event):
        print('Quit')
        self.Close()

    def save_enable(self):
        for child in self.gridSaveSizer.GetChildren():
            child.GetWindow().Enable()


if __name__ == '__main__':
    app = wx.App()
    ex = Panel(None)
    ex.Show()
    app.MainLoop()

