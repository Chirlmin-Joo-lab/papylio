""""
This scripts collects a number of standard example scripts.
It is intended for:
1) GUI users that wish to use small custom functions from with the GUI framework
2) GUI users that wish to gain more insight in script-based Papylio use

The scripts are organized in Themes, as such visible in the GUI:
example_scripts = {
    "Show"
    "Select"
    "Analyze"
}
"""


example_scripts_show = {
    "Hello World": """
#edit this script at wish:
print("Hello world")
""",
    "Show mapping": """
#pick your mapping file:
mapping_file = experiment.files[0]
#show it:
figure, axis = mapping_file.show_image()
mapping_file.mapping.show(axis=axis, show_source=True)
figure.show()
""",

    "Show Average Image": """
#show the average image of the selected file
file.show_average_image()
self.update_plots()
""",

    "Show 1D Histogram": """
#Plot the histogram:
fig, ax = file.show_histogram('intensity', selected=True, bins=100)
fig.show()
""",

    "Show 2D Histogram": """
#show a 2-D histogram of the selected file, with a user-set range
import matplotlib.pyplot as plt
kwargs = {'range': ((-10E3, 30E3), (-5E3, 15E3))}
file.histogram_2D_intensity_per_channel(    bins=100, show_marginal=False,**kwargs)
plt.show()

""",
}

example_scripts_select = {
    "Show selected files": """
for file in experiment.selectedFiles:
    print(file)
""",

    "Create Trace Selection": """
#Create a selection that will be added to the list of selection rules - see 'Selection' Tab:
file.create_selection(variable='intensity', channel='red', aggregator='max', operator='>', threshold=3000, name='selection_red')
file.apply_selections()
""",
}

example_scripts_analyze= {
"Estimate_psf": """
#Determine the psf size based on the detected spots (may take ~20s)
psf=file.determine_psf_size(method='gaussian_fit', projection_type='average', frame_range=(0,10), channel_index=0, illumination_index=0,
                           peak_finding_kwargs={'minimum_intensity_difference': 150}, maximum_radius=5)
self.update_plots()
print(psf)
    """,

"Dwell times": """
#Get dwell times , analyze and show the result:
file.determine_dwells_from_classification(variable='FRET', selected=True, inactivate_start_and_end_states=True)
file.analyze_dwells(method='histogram_fit', number_of_exponentials=[1, 2])
fig,ax=file.plot_dwell_analysis(plot_range=[(0,2),(0,4)], log=False, save_path=None)
fig.show()
    """,
}

example_scripts_process_traces= {
"median_FRET": """
#Build a variable that comprises of a median-filtered FRET
fret=file.get_variable("FRET")
fret_median5 = fret.rolling(frame=5, center=True, min_periods=1).median()
file.set_variable(fret_median5, name='fret_median_5')
    """,

}

example_scripts = {
    "Show": example_scripts_show,
    "Select": example_scripts_select,
    "Analyze": example_scripts_analyze,
    "Pre_process": example_scripts_process_traces,
}