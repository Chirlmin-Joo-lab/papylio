import papylio as pp

# Experiment import
exp = pp.Experiment()
exp.print_files()


# Emission channel mapping
mapping_file_index = 0
mapping_file = exp.files[mapping_file_index]

mapping_file.perform_mapping()

figure, axis = mapping_file.show_image()
mapping_file.mapping.show(axis=axis, show_source=True)


# File selection
file_selection = exp.files[2:10]  # Selection of files: change to the desired file range


# Background subtraction
file_selection.movie.determine_spatial_background_correction()


# Molecule localization - test on one file
test_file = file_selection[0]  # Change number to the desired file number
test_file.find_coordinates()
test_file.show_coordinates_in_image()


# Molecule localization - apply to all files
file_selection.find_coordinates()


# Trace extraction
file_selection.extract_traces()


# Show traces
test_file = file_selection[0]  # Change number to the desired file number
test_file.show_traces(plot_variables=['intensity', 'FRET'],
                     ylims=[(0, 35000), (0, 1)],
                     colours=[('green', 'red'), ('blue')])
