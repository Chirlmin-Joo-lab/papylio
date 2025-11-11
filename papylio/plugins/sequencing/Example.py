import numpy as np
from pathlib import Path # For efficient path manipulation
import papylio as pp
import matplotlib.pyplot as plt
import tqdm
# from git import Repo
import time
import xarray as xr
from skimage.transform import AffineTransform

from papylio.plugins.sequencing.fastqAnalysis import FastqData
import matchpoint as mp

# This part makes an analysis.txt file in the same directory as the this .py file. (please do not commit this)
# This file contains the git commit used and all differences to this commit.
# writePath = Path(Path(__file__).parent)
# writeFile = writePath.joinpath('git_repository_version_and_differences.txt')
# repo = Repo(Path(pp.__file__).parent.parent)
#
# with writeFile.open("a") as f:
#     f.write('------------------------------------------\n\n')
#     f.write(f'Trace_analysis version: {repo.head.object.hexsha} \n\n')
#     t = repo.head.commit.tree
#     f.write(repo.git.diff(t))
#     f.write('\n\n------------------------------------------\n\n')



#####################################
# SINGLE-MOLECULE DATA PROCESSING
#####################################

experiment_path = r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20221023 - Objective-type TIRF (BN)'
# experiment_path = r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20221023 - Objective-type TIRF (BN)\Test'
exp = pp.Experiment(experiment_path)

files_channel_mapping = exp.files[exp.files.relativeFilePath.str.regex('Bead slide')]
files_green_laser = exp.files[exp.files.relativeFilePath.str.regex('3.*HJ_scan TIRF 561')]
files_red_laser_after = exp.files[exp.files.relativeFilePath.str.regex('3.*HJ_scan TIRF 642 after')]
files_red_laser_before = exp.files[exp.files.relativeFilePath.str.regex('3.*HJ_scan TIRF 642 before')]

files_red_laser_before.movie.illumination_arrangement = [1]
files_red_laser_after.movie.illumination_arrangement = [1]

# -----------------------------------
# Channel mapping
# -----------------------------------
# channel_mapping_file = files_channel_mapping[0]
# channel_mapping_file.perform_mapping()
#
# channel_mapping_file.show_average_image()
# channel_mapping_file.mapping.show_mapping_transformation(figure=plt.gcf(), show_source=True)

# -----------------------------------
# Image corrections
# -----------------------------------
exp.determine_flatfield_and_darkfield_corrections(files_green_laser, illumination_index=0, l_s=5, l_d=5)
exp.determine_flatfield_and_darkfield_corrections(files_red_laser_before, illumination_index=1, l_s=5, l_d=5)

files_green_laser.movie.determine_temporal_background_correction('fit_background_peak')
files_red_laser_before.movie.determine_temporal_background_correction('fit_background_peak')
files_red_laser_after.movie.determine_temporal_background_correction('fit_background_peak')

for file in tqdm.tqdm(files_green_laser):
    temporal_illumination_correction = file.movie.temporal_background_correction.sel(channel=1) / 1500
    temporal_background_correction = file.movie.temporal_background_correction / temporal_illumination_correction
    file.movie.temporal_background_correction = temporal_background_correction
    file.movie.temporal_illumination_correction = temporal_illumination_correction
    file.movie.save_corrections('temporal_background_correction', 'temporal_illumination_correction')

# -----------------------------------
# Find coordinates, extract traces and determine kinetics
# -----------------------------------

files_green_laser.find_coordinates()
files_green_laser[652].show_coordinates_in_image()

files_green_laser.extract_traces()


def add_coordinates_to_files(file_green, file_red_before, file_red_after):
    if (file_green.name[-4:] == file_red_before.name[-4:] == file_red_after.name[-4:]) and \
            (file_green.relativePath.name == file_red_before.relativePath.name == file_red_after.relativePath.name):
        file_red_before.coordinates = file_green.coordinates
        file_red_after.coordinates = file_green.coordinates
    else:
        raise FileNotFoundError

# TODO: Make this internal somehow
import joblib
from objectlist.base import tqdm_joblib
# for file_green, file_red_before, file_red_after in
with tqdm_joblib(tqdm.tqdm(total=len(files_green_laser))):
    joblib.parallel.Parallel(6)(joblib.parallel.delayed(add_coordinates_to_files)(*datum) for datum in
                                zip(files_green_laser, files_red_laser_before, files_red_laser_after))

files_red_laser_before.extract_traces()
files_red_laser_after.extract_traces()

def add_red_laser_intensity_to_files_green(file_green, file_red_before, file_red_after):
    if (file_green.name[-4:] == file_red_before.name[-4:] == file_red_after.name[-4:]) and \
            (file_green.relativePath.name == file_red_before.relativePath.name == file_red_after.relativePath.name):
        intensity_red_before = file_red_before.intensity.mean('frame')
        intensity_red_before = intensity_red_before.drop('file')
        intensity_red_before.name = 'intensity_red_before'
        intensity_red_before.to_netcdf(file_green.absoluteFilePath.with_suffix('.nc'), engine='h5netcdf', mode='a')

        intensity_red_after = file_red_after.intensity.mean('frame')
        intensity_red_after = intensity_red_after.drop('file')
        intensity_red_after.name = 'intensity_red_after'
        intensity_red_after.to_netcdf(file_green.absoluteFilePath.with_suffix('.nc'), engine='h5netcdf', mode='a')

    else:
        raise FileNotFoundError


with tqdm_joblib(tqdm.tqdm(total=len(files_green_laser))):
    joblib.parallel.Parallel(6)(joblib.parallel.delayed(add_red_laser_intensity_to_files_green)(*datum) for datum in
                                zip(files_green_laser, files_red_laser_before, files_red_laser_after))




#####################################
# IMPORT SEQUENCING DATA
#####################################

sequencing_analysis_path = Path(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20221024 - Sequencer (MiSeq)\Analysis')
aligned_sam_filepath = sequencing_analysis_path.joinpath('Alignment_read1.sam')
index1_fastq_filepath = sequencing_analysis_path.joinpath('Index1.fastq') # Can be None
extract_sequence_subset = [30, 31, 56, 57, 82, 83, 108, 109]

exp.import_sequencing_data(aligned_sam_filepath, index1_file_path=index1_fastq_filepath, remove_duplicates=True,
                           add_aligned_sequence=True, extract_sequence_subset=extract_sequence_subset)

#####################################
# SEQUENCING AND SINGLE-MOLECULE MAPPING
#####################################

mapping_sequence_name = 'HJ_general'

# files_scan123 = files_red_laser_before[files_red_laser_before.relativePath.str.regex('Scan 1|Scan 3')]
exp.generate_tile_mappings(files_green_laser, mapping_sequence_name=mapping_sequence_name, surface=0)

# files_scan4 = files_red_laser_before[files_red_laser_before.relativePath.str.regex('Scan 4')]
# exp.generate_tile_mappings(files_scan4, mapping_sequence_name=mapping_sequence_name, surface=0, name='Scan 4')


# -----------------------------------
# Finding rotation and scale with geometric hashing
# -----------------------------------

# TODO: Geometric hashing example here
# TODO: Put geometric hashing in MatchPoint



# -----------------------------------
# Finding translation with cross correlation
# -----------------------------------

# Previously found transformation
# Transformation based on sm pixel to sequencing MiSeq mapping with
# 'rotation': 0.006500218506032994, 'scale': [3.697153992993506, -3.697153992993506] using pixel size 0.125 µm
# exp.tile_mappings.transformation = AffineTransform(scale=[29.57723194394805, -29.57723194394805], rotation=0.006500218506032994)
exp.tile_mappings.transformation = AffineTransform(scale=[29.51, -29.51], rotation=0.0024)

exp.tile_mappings.use_parallel_processing = False
exp.tile_mappings.cross_correlation(peak_detection='auto', gaussian_width=7, divider=20, plot=False)

# bounds = ((0.98, 1.2), (-0.005, 0.005), (-250, 250), (-250, 250))
# exp.tile_mappings[19:(19+12)].kernel_correlation(bounds, sigma=25, crop='source',
#                                      strategy='best1bin', maxiter=1000, popsize=50, tol=0.01,
#                                      mutation=0.25, recombination=0.7, seed=None, callback=None, disp=False,
#                                      polish=False, init='sobol', atol=0, updating='immediate', workers=1,
#                                      constraints=())

exp.tile_mappings.save()
exp.tile_mappings.show_mapping_transformation(crop='source', save=True)

# -----------------------------------
# Determine translations for all tiles
# -----------------------------------
exp.tile_mappings.scatter_parameters('translation', 'translation', 'x', 'y', save=True)  # To determine correct mapping indices
exp.tile_mappings.estimate_translations(indices=np.arange(11), save=True)
exp.tile_mappings.scatter_parameters('translation', 'translation', 'x', 'y', save=True) # Rename original file before saving
exp.tile_mappings.save()
exp.tile_mappings.show_mapping_transformation(crop='source', save=True)

# -----------------------------------
# Matching statistics
# -----------------------------------

# TODO: Show how to extract matching statics, i.e. numbers/percentages of points matched.

# -----------------------------------
# Obtain file sequencing data and generate sequencing match
# -----------------------------------
files = files_green_laser[0:14*30]

# TODO: Make this automatically save the sequencing data
# TODO: Automatic import and export when setting or getting sequencing data
files_green_laser.parallel_processing_kwargs['require'] = 'sharedmem'
# files_green_scan123 = files_green_laser[files_green_laser.relativePath.str.regex('Scan 1|Scan 3')]
files_green_laser.get_sequencing_data(margin=5, mapping_name='All files')
# files_green_scan4 = files_green_laser[files_green_laser.relativePath.str.regex('Scan 4')]
# files_green_scan4.get_sequencing_data(margin=5, mapping_name='Scan 4 - HJ general')

files_green_laser.parallel_processing_kwargs.pop('require')
files_green_laser.generate_sequencing_match(overlapping_points_threshold=25,
                                            excluded_sequence_names=['MapSeq', 'CalSeq', '*'], plot=False)

# -----------------------------------
# Finetune the sequencing matches
# -----------------------------------

sequencing_matches = exp.sequencing_matches(files_green_laser)

sequencing_matches.cross_correlation(divider=1/10, gaussian_width=7, crop=True, plot=False)


sequencing_matches.transformation = AffineTransform()
sequencing_matches.transformation_inverse = AffineTransform()
bounds = ((0.97, 1.03), (-0.05, 0.05), (-5, 5), (-5, 5))
sequencing_matches.kernel_correlation(bounds, sigma=0.125, crop=True,
                                         strategy='best1bin', maxiter=1000, popsize=50, tol=0.01,
                                         mutation=0.25, recombination=0.7, seed=None, callback=None, disp=False,
                                         polish=True, init='sobol', atol=0, updating='immediate', workers=1,
                                         constraints=())

bounds = ((0.99, 1.01), (-0.01, 0.01), (-1, 1), (-1, 1))
sequencing_matches.kernel_correlation(bounds, sigma=0.125, crop=True,
                                         strategy='best1bin', maxiter=1000, popsize=50, tol=0.001,
                                         mutation=0.25, recombination=0.7, seed=None, callback=None, disp=False,
                                         polish=True, init='sobol', atol=0, updating='immediate', workers=1,
                                         constraints=())
sequencing_matches.save()

# -----------------------------------
# Find pairs and insert sequencing data into file dataset
# -----------------------------------

sequencing_matches.find_distance_threshold(maximum_radius=2)
sequencing_matches.determine_matched_pairs()
plt.figure()
plt.hist(np.hstack(sequencing_matches.pair_distances()), bins=100)
plt.title('Pair distances for istance_threshold 0.42 um')
plt.xlabel('Distance (um)')
plt.ylabel('Count')

sequencing_matches.destination_distance_threshold = 0.2  # 0.506
sequencing_matches.determine_matched_pairs()
sequencing_matches.save()

sequencing_matches.show_mapping_transformation()

files_green_laser.insert_sequencing_data_into_file_dataset(include_raw_sequences=False, include_aligned_sequences=True,
                                                           include_sequence_subset=True, determine_matched_pairs=True)


source_cropped_lengths = sequencing_matches.source_cropped.map(len)()
destination_cropped_lengths = sequencing_matches.destination_cropped.map(len)()
matched_pairs_lenghts = sequencing_matches.number_of_matched_points


source_transformed_combined = np.vstack([sequencing_match.transformation(sequencing_match.source) for sequencing_match in sequencing_matches])
destination_combined = np.unique(np.vstack([sequencing_match.destination for sequencing_match in sequencing_matches]), axis=0)
test = mp.MatchPoint(source=source_transformed_combined, destination=destination_combined)
test.destination_distance_threshold = 0.2
test.determine_pairs()

# ds = xr.open_mfdataset([file.relativeFilePath.with_suffix('.nc') for file in files_green_laser[0:300]], combine='nested', concat_dim='molecule',
#                        data_vars='minimal', coords='minimal', compat='override', engine='h5netcdf')#, parallel=True)#, chunks={'molecule': 100})
import time
start = time.time()
ds = xr.open_mfdataset([file.relativeFilePath.with_suffix('.nc') for file in files_green_laser], combine='nested', concat_dim='molecule',
                       data_vars='minimal', coords='minimal', compat='override', engine='h5netcdf', parallel=False)
print(time.time()-start)


# dataset_complete_subset = ds.sel(molecule=~ds.sequence_subset.str.contains(b'-'))
#
# folder = exp.main_path.joinpath('Analysis').joinpath('complete_dataset.nc')
# for group, value in tqdm.tqdm(ds.groupby('sequence_subset')):
#     if b'-' not in group and group != b'':
#         value.to_netcdf(folder.joinpath(group.decode()).with_suffix('.nc'), engine='h5netcdf')





import time
start = time.time()
ds.to_netcdf(exp.main_path.joinpath('Analysis').joinpath('complete_dataset.nc'), engine='h5netcdf', mode='w')
print(time.time()-start)



stage_coordinates = files_green_laser[(38*30):((38+15)*30)].movie.stage_coordinates

tile = exp.sequencing_data.sel(tile=1104)
ch = scipy.spatial.ConvexHull(tile.coordinates, incremental=False, qhull_options=None)


hull_coordinates = tile.coordinates[ch.vertices]
hull_coordinates = np.vstack([hull_coordinates.values, hull_coordinates.values[0]])
hull_coordinates_um = exp.tile_mappings[np.where((np.array(exp.tile_mappings.tile)==1101))[0].item()].transform_coordinates(hull_coordinates, inverse=True)

plt.figure()
hull_coordinates_um[:, 1] += 3200
hull_coordinates_um[:, 0] += 40
plt.plot(*hull_coordinates_um.T, c='r')
outline = np.array([[0, 0], [0, 64], [32, 64], [32, 0], [0, 0]])
stage_coordinates_adapted = np.flip(np.vstack(stage_coordinates), axis=1)
stage_coordinates_adapted[:,1] += 3200
for sc in stage_coordinates_adapted:
    plt.plot(*(sc + outline).T, c='g')
plt.gca().set_aspect('equal', 'box')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.tight_layout()



axis('equal')


ds = xr.open_dataset(exp.main_path.joinpath('Analysis').joinpath('complete_dataset.nc'), engine='h5netcdf')#, chunks={'molecule': 2000})




sequence_subset_hj1 = b'TTAGCCGA' # HJ1
sequence_subset_hj3 = b'AATCGGCT' # HJ3
sequence_subset_hj7 = b'GGCGCCGC' # HJ7






sequence_mutations_hj7 = ['GGCGCCGC', 'AGCGCCGC', 'CGCGCCGC', 'TGCGCCGC', 'AACGCCGC', 'ACCGCCGC', 'ATCGCCGC', 'CACGCCGC', 'CCCGCCGC', 'CTCGCCGC', 'TACGCCGC', 'TCCGCCGC', 'TTCGCCGC', 'AGAGCCGC', 'AGTGCCGC', 'AGGGCCGC', 'CGAGCCGC', 'CGTGCCGC', 'CGGGCCGC', 'TGAGCCGC', 'TGTGCCGC', 'TGGGCCGC', 'AGCACCGC', 'AGCCCCGC', 'AGCTCCGC', 'CGCACCGC', 'CGCCCCGC', 'CGCTCCGC', 'TGCACCGC', 'TGCCCCGC', 'TGCTCCGC', 'AGCGACGC', 'AGCGTCGC', 'AGCGGCGC', 'CGCGACGC', 'CGCGTCGC', 'CGCGGCGC', 'TGCGACGC', 'TGCGTCGC', 'TGCGGCGC', 'AGCGCAGC', 'AGCGCTGC', 'AGCGCGGC', 'CGCGCAGC', 'CGCGCTGC', 'CGCGCGGC', 'TGCGCAGC', 'TGCGCTGC', 'TGCGCGGC', 'AGCGCCAC', 'AGCGCCCC', 'AGCGCCTC', 'CGCGCCAC', 'CGCGCCCC', 'CGCGCCTC', 'TGCGCCAC', 'TGCGCCCC', 'TGCGCCTC', 'AGCGCCGA', 'AGCGCCGT', 'AGCGCCGG', 'CGCGCCGA', 'CGCGCCGT', 'CGCGCCGG', 'TGCGCCGA', 'TGCGCCGT', 'TGCGCCGG', 'GACGCCGC', 'GCCGCCGC', 'GTCGCCGC', 'GAAGCCGC', 'GATGCCGC', 'GAGGCCGC', 'GCAGCCGC', 'GCTGCCGC', 'GCGGCCGC', 'GTAGCCGC', 'GTTGCCGC', 'GTGGCCGC', 'GACACCGC', 'GACCCCGC', 'GACTCCGC', 'GCCACCGC', 'GCCCCCGC', 'GCCTCCGC', 'GTCACCGC', 'GTCCCCGC', 'GTCTCCGC', 'GACGACGC', 'GACGTCGC', 'GACGGCGC', 'GCCGACGC', 'GCCGTCGC', 'GCCGGCGC', 'GTCGACGC', 'GTCGTCGC', 'GTCGGCGC', 'GACGCAGC', 'GACGCTGC', 'GACGCGGC', 'GCCGCAGC', 'GCCGCTGC', 'GCCGCGGC', 'GTCGCAGC', 'GTCGCTGC', 'GTCGCGGC', 'GACGCCAC', 'GACGCCCC', 'GACGCCTC', 'GCCGCCAC', 'GCCGCCCC', 'GCCGCCTC', 'GTCGCCAC', 'GTCGCCCC', 'GTCGCCTC', 'GACGCCGA', 'GACGCCGT', 'GACGCCGG', 'GCCGCCGA', 'GCCGCCGT', 'GCCGCCGG', 'GTCGCCGA', 'GTCGCCGT', 'GTCGCCGG', 'GGAGCCGC', 'GGTGCCGC', 'GGGGCCGC', 'GGAACCGC', 'GGACCCGC', 'GGATCCGC', 'GGTACCGC', 'GGTCCCGC', 'GGTTCCGC', 'GGGACCGC', 'GGGCCCGC', 'GGGTCCGC', 'GGAGACGC', 'GGAGTCGC', 'GGAGGCGC', 'GGTGACGC', 'GGTGTCGC', 'GGTGGCGC', 'GGGGACGC', 'GGGGTCGC', 'GGGGGCGC', 'GGAGCAGC', 'GGAGCTGC', 'GGAGCGGC', 'GGTGCAGC', 'GGTGCTGC', 'GGTGCGGC', 'GGGGCAGC', 'GGGGCTGC', 'GGGGCGGC', 'GGAGCCAC', 'GGAGCCCC', 'GGAGCCTC', 'GGTGCCAC', 'GGTGCCCC', 'GGTGCCTC', 'GGGGCCAC', 'GGGGCCCC', 'GGGGCCTC', 'GGAGCCGA', 'GGAGCCGT', 'GGAGCCGG', 'GGTGCCGA', 'GGTGCCGT', 'GGTGCCGG', 'GGGGCCGA', 'GGGGCCGT', 'GGGGCCGG', 'GGCACCGC', 'GGCCCCGC', 'GGCTCCGC', 'GGCAACGC', 'GGCATCGC', 'GGCAGCGC', 'GGCCACGC', 'GGCCTCGC', 'GGCCGCGC', 'GGCTACGC', 'GGCTTCGC', 'GGCTGCGC', 'GGCACAGC', 'GGCACTGC', 'GGCACGGC', 'GGCCCAGC', 'GGCCCTGC', 'GGCCCGGC', 'GGCTCAGC', 'GGCTCTGC', 'GGCTCGGC', 'GGCACCAC', 'GGCACCCC', 'GGCACCTC', 'GGCCCCAC', 'GGCCCCCC', 'GGCCCCTC', 'GGCTCCAC', 'GGCTCCCC', 'GGCTCCTC', 'GGCACCGA', 'GGCACCGT', 'GGCACCGG', 'GGCCCCGA', 'GGCCCCGT', 'GGCCCCGG', 'GGCTCCGA', 'GGCTCCGT', 'GGCTCCGG', 'GGCGACGC', 'GGCGTCGC', 'GGCGGCGC', 'GGCGAAGC', 'GGCGATGC', 'GGCGAGGC', 'GGCGTAGC', 'GGCGTTGC', 'GGCGTGGC', 'GGCGGAGC', 'GGCGGTGC', 'GGCGGGGC', 'GGCGACAC', 'GGCGACCC', 'GGCGACTC', 'GGCGTCAC', 'GGCGTCCC', 'GGCGTCTC', 'GGCGGCAC', 'GGCGGCCC', 'GGCGGCTC', 'GGCGACGA', 'GGCGACGT', 'GGCGACGG', 'GGCGTCGA', 'GGCGTCGT', 'GGCGTCGG', 'GGCGGCGA', 'GGCGGCGT', 'GGCGGCGG', 'GGCGCAGC', 'GGCGCTGC', 'GGCGCGGC', 'GGCGCAAC', 'GGCGCACC', 'GGCGCATC', 'GGCGCTAC', 'GGCGCTCC', 'GGCGCTTC', 'GGCGCGAC', 'GGCGCGCC', 'GGCGCGTC', 'GGCGCAGA', 'GGCGCAGT', 'GGCGCAGG', 'GGCGCTGA', 'GGCGCTGT', 'GGCGCTGG', 'GGCGCGGA', 'GGCGCGGT', 'GGCGCGGG', 'GGCGCCAC', 'GGCGCCCC', 'GGCGCCTC', 'GGCGCCAA', 'GGCGCCAT', 'GGCGCCAG', 'GGCGCCCA', 'GGCGCCCT', 'GGCGCCCG', 'GGCGCCTA', 'GGCGCCTT', 'GGCGCCTG', 'GGCGCCGA', 'GGCGCCGT', 'GGCGCCGG']


molecule_count = {s: ((ds.sequence_subset == s.encode()) & (ds.sequence_name == b'HJ_general')).sum().item() for s in sequence_mutations_hj7}
mc = xr.DataArray(list(molecule_count.values()), dims=('sequence'), coords={'sequence': list(molecule_count.keys())})
mc.to_netcdf(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\HJ7 mutations\molecule_count.nc')


for seq in sequence_mutations_hj7:
    seq_sel = ((ds.sequence_subset == seq.encode()) & (ds.sequence_name == b'HJ_general'))
    trace_data = ds[['intensity_red_before', 'intensity', 'FRET']].sel(molecule=seq_sel)
    if len(trace_data.molecule) > 0:
        trace_data.to_netcdf(rf'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\Datasets per sequence\{seq}.nc')

from papylio.analysis.classification_simple import classify_anticorrelation



da = xr.DataArray(0, coords={'sequence': sequence_mutations_hj7 })

fig, ax = plt.subplots()




for seq in tqdm.tqdm(sequence_mutations_hj7):
    filepath = Path(rf'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\Datasets per sequence\{seq}.nc')
    if filepath.is_file():
        ds_seq = xr.load_dataset(filepath)
        ds_seq.close()
        ds_seq['selected'] = xr.DataArray(False, dims=('molecule',), coords={'molecule': ds_seq.molecule.values}).drop_vars('molecule')
        ds_seq['classification_anticorrelation'] = classify_anticorrelation(ds_seq.intensity, window=20, rolling_mean_window=10, threshold=-0.5)
        ds_seq.to_netcdf(rf'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\Datasets per sequence\{seq}.nc')


counts = []
anticorrelation_values = []
for seq in tqdm.tqdm(sequence_mutations_hj7):
    filepath = Path(rf'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\Datasets per sequence\{seq}.nc')
    if filepath.is_file():
        ds_seq = xr.load_dataset(filepath)
        anticorrelation_with_red = ds_seq['classification_anticorrelation'].sel(molecule=(ds_seq.intensity_red_before.sel(channel=1)>2000))
        if len(anticorrelation_with_red.molecule)>0:
            anticorrelation_value = (anticorrelation_with_red.sum() / anticorrelation_with_red.size).item()
            anticorrelation_values.append(anticorrelation_value)
            counts.append(len(anticorrelation_with_red.molecule))
        else:
            anticorrelation_values.append(np.nan)
            counts.append(0)
            print('No red')
    else:
        anticorrelation_values.append(np.nan)
        counts.append(0)

cc = xr.DataArray(counts, dims=('sequence'), coords={'sequence': sequence_mutations_hj7}, name='Molecule count')
ac = xr.DataArray(anticorrelation_values, dims=('sequence'), coords={'sequence': sequence_mutations_hj7}, name='Fraction showing anticorrelation')
eds = ac.to_dataset()
eds['Molecule count'] = cc
eds.to_netcdf(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\HJ7 mutations\fraction_anticorrelation.nc')


from papylio.trace_plot import TracePlotWindow
from papylio.experiment import get_QApplication
app = get_QApplication()
frame = TracePlotWindow(ds_seq, plot_variables=['intensity', 'FRET', 'classification_anticorrelation'],
                        ylims=[(0, 15000), (-0.1,1.1), (-1,2)], colours=[('g', 'r'), ('b'), ('k')], save_path=None)
app.exec_()







for seq in sequence_mutations_hj7:
    seq_sel = ((ds.sequence_variable == seq.encode()))
    trace_data = ds[['intensity_red_laser', 'intensity', 'FRET', 'classification']].sel(molecule=seq_sel)
    if len(trace_data.molecule) > 0:
        trace_data.to_netcdf(exp.main_path.joinpath('Analysis').joinpath('HJ7 mutations').joinpath(seq).with_suffix('.nc'))


counts = []
anticorrelation_values = []
for seq in tqdm.tqdm(sequence_mutations_hj7):
    filepath = exp.main_path.joinpath('Analysis').joinpath('HJ7 mutations').joinpath(seq).with_suffix('.nc')
    if filepath.is_file():
        ds_seq = xr.load_dataset(filepath)
        anticorrelation_trace = ds_seq['classification']>=0
        if len(anticorrelation_trace.molecule)>0:
            anticorrelation_value = (anticorrelation_trace.sum() / anticorrelation_trace.size).item()
            anticorrelation_values.append(anticorrelation_value)
            counts.append(len(anticorrelation_trace.molecule))
        else:
            anticorrelation_values.append(np.nan)
            counts.append(0)
    else:
        anticorrelation_values.append(np.nan)
        counts.append(0)


cc = xr.DataArray(counts, dims=('sequence'), coords={'sequence': sequence_mutations_hj7}, name='Molecule count')
ac = xr.DataArray(anticorrelation_values, dims=('sequence'), coords={'sequence': sequence_mutations_hj7}, name='Fraction showing anticorrelation')
eds = ac.to_dataset()
eds['Molecule count'] = cc
eds.to_netcdf(exp.main_path.joinpath('Analysis').joinpath('HJ7 mutations').joinpath('fraction_anticorrelation.nc'))
# eds.to_netcdf(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20220602 - Objective-type TIRF (BN)\Analysis\HJ7 mutations\fraction_anticorrelation.nc')


# ds = xr.open_mfdataset([file.relativeFilePath.with_suffix('.nc') for file in files_green_laser[0:300]], combine='nested', concat_dim='molecule',
#                        data_vars='minimal', coords='minimal', compat='override', engine='h5netcdf')#, parallel=True)#, chunks={'molecule': 100})
import time
start = time.time()
ds = xr.open_mfdataset([file.relativeFilePath.with_suffix('.nc') for file in files_green_laser], combine='nested', concat_dim='molecule',
                       data_vars='minimal', coords='minimal', compat='override', engine='h5netcdf', parallel=False)
print(time.time()-start)

import time
start = time.time()
ds.to_netcdf(r'N:\tnw\BN\CMJ\Shared\Ivo\PhD_data\20211005 - Objective-type TIRF (BN)\complete_dataset.nc', engine='h5netcdf')
print(time.time()-start)

def read_netcdfs(files, dim, transform_func=None):
    paths = files #sorted(glob(files))
    datasets = []
    for path in paths:
        with xr.open_dataset(path, engine='h5netcdf') as ds:
            if transform_func is not None:
                ds = transform_func(ds)
            if len(ds.molecule)>0:
                ds.load()
                datasets.append(ds)
    combined = xr.concat(datasets, dim)
    return combined

def ff(ds):
    return ds.sel(molecule=(ds.sequence_variable == 'CAGCAGCA'))

filepaths = files_green_laser.absoluteFilePath.with_suffix('.nc')
start = time.time()
test = read_netcdfs(filepaths, 'molecule', transform_func=ff)
print(time.time()-start)


for file in tqdm.tqdm(files_green_laser[400:]):
    ds = file.dataset

    # s = ds['sequence']==''
    # ds['sequence'][s]='-'*120
    # ds['sequence_quality'][s]=' '*120
    # ds = ds.drop_vars('distance_to_sequence', errors='ignore')
    # for i, index in enumerate([30,31,56,57,82,83,108,109]):
    #     if i == 0:
    #         sequence_variable = ds.sequence.str.get(index)
    #     else:
    #         sequence_variable += ds.sequence.str.get(index)
    #
    # ds['sequence_variable'] = sequence_variable

    # keys = ['file','dimension','sequence','sequence_quality','sequence_variable']
    # for key in keys:
    #     ds[key] = ds[key].astype('|S')

    encoding = {'file': {'dtype': '|S'}, 'dimension': {'dtype': '|S'}, 'selected': {'dtype': bool}}
    ds.to_netcdf(file.absoluteFilePath.with_suffix('.nc'), engine='h5netcdf', mode='w', encoding=encoding)


def change_type(file):
    ds = file.dataset
    encoding = {'file': {'dtype': '|S'}, 'dimension': {'dtype': '|S'}, 'selected': {'dtype': bool}}
    ds.to_netcdf(file.absoluteFilePath.with_suffix('.nc'), engine='h5netcdf', mode='w', encoding=encoding)

import joblib
from objectlist.base import tqdm_joblib
# for file_green, file_red_before, file_red_after in
with tqdm_joblib(tqdm.tqdm(total=len(files_green_laser[400:]))):
    joblib.parallel.Parallel(6)(joblib.parallel.delayed(change_type)(file) for file in files_green_laser[400:])

# -----------------------------------
# Select sequencing data for mapping
# -----------------------------------
mapping_sequence = 'ACTGACTGTAACAACAACAACAATAACAACAACAACAATAACAACAACAACAAT'
number_of_allowed_mismatches = 2
exp.select_sequencing_data_for_mapping(mapping_sequence, number_of_allowed_mismatches)

# -----------------------------------
# Find high-intensity coordinates
# -----------------------------------
configuration = exp.configuration['find_coordinates'].copy()
configuration['peak_finding']['minimum_intensity_difference'] = 1500

for file in files_red_laser:
    file.find_coordinates(configuration=configuration)


# -----------------------------------
# Geometric hashing 2.0
# -----------------------------------
# # This part is likely not yet working well
# start = time.time()
# exp.generate_mapping_hashtable(mapping_sequence, number_of_allowed_mismatches,
#                                imaged_surface=2, maximum_distance_tile=3000, tuple_size=4)
# end = time.time()
# print(f'Hashtable generation: {end - start} seconds')
#
# # Or just use the coordinates from the loc files
# # path_to_loc_folder = Path(r'Path_to_loc_folder')
# # tile_coordinate_files = ['2101.loc', '2102.loc', '2103.loc', '2104.loc']
# # tile_coordinate_sets = [np.loadtxt(path_to_loc_folder.joinpath(f)) for f in tile_coordinate_files]
# # start = time.time()
# # exp.generate_mapping_hashtable_from_coordinate_set(tile_coordinate_sets, maximum_distance_tile=3000, tuple_size=4)
# # end = time.time()
# # print(f'Hashtable generation: {end - start} seconds')
#
# # For each file in experiment find a match
# # NOTE: 'scale': [-1,1] means a reflection with respect to the y axis.
#
# start = time.time()
# for file in exp.files:
#     file.sequencing_match = None
#     file.find_sequences(maximum_distance_file=1000, tuple_size=4, initial_transformation={'scale': [-1,1]},
#                         hash_table_distance_threshold=0.01,
#                         alpha=0.1, test_radius=10, K_threshold=10e6, # original K_threshold = 10e9
#                         magnification_range=[3.3,3.4], rotation_range=[-1,1])
#     print(file)
# end = time.time()
# print(f'Matching: {end - start} seconds')
#
# matched_files = [file for file in exp.files if file.sequencing_match]
# number_of_matches = len(matched_files)
#
# # Improve mapping by performing a linear least-squares fit on all nearest neighbours within the distance threshold
# for file in matched_files:
#     nearest_neighbour_match_distance_threshold = 25
#     file.sequencing_match.nearest_neighbour_match(nearest_neighbour_match_distance_threshold)
#
# # Make and export a sequencing match plot for files that have a sequencing match
# for file in matched_files: #exp.files # files_correct: #exp.files:
#     if file.sequencing_match:
#         file.plot_sequencing_match()
#         # plt.close()

# -----------------------------------
# ASSESSING OVERALL MAPPING QUALITY
# -----------------------------------
#
# # Calculate rotation and magnification for all found matches
# rotation = np.array([file.sequencing_match.rotation for file in exp.files if file.sequencing_match])
# magnification = np.array([file.sequencing_match.magnification for file in exp.files if file.sequencing_match])
# mean_magnification = np.mean(magnification, axis=1)
#
# # Make a heatmap scatter plot
# from scipy.stats import gaussian_kde
# x = rotation
# y = mean_magnification
#
# xy = np.vstack([x,y])
# z = gaussian_kde(xy,0.01)(xy)
# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=100, edgecolor='')
# plt.title('Matches')
# plt.xlabel('Rotation (deg)')
# plt.ylabel('Magnfication (average over two directions)')
# plt.show()
#
# # Make a heatmap scatter plot
# from scipy.stats import gaussian_kde
# selection = ((rotation>179) | (rotation<-179)) & (mean_magnification>3.3) & (mean_magnification<3.4)
# x = rotation[selection]
# y = mean_magnification[selection]
# xy = np.vstack([x,y])
# z = gaussian_kde(xy,0.01)(xy)
# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=100, edgecolor='')
# plt.title('Matches')
# plt.xlabel('Rotation (deg)')
# plt.ylabel('Magnfication (average over two directions)')
# plt.show()


# -----------------------------------
# Geometric hashing 3.0
# -----------------------------------

# Define initial file transformation
initial_magnification = np.array([3.71, -3.71])
initial_rotation = 0.4

initial_file_transformation = AffineTransform(matrix=None, scale=initial_magnification,
                                                rotation=initial_rotation/360*np.pi*2,
                                                shear=None, translation=None)

# TODO: Add timer to generate_mapping_hashtable and find_sequences methods, by making a decorator function. [IS: 10-08-2020]
start = time.time()
exp.generate_mapping_hashtable3(mapping_sequence, number_of_allowed_mismatches,
                               imaged_surface=1, initial_file_transformation=initial_file_transformation)
end = time.time()
print(f'Hashtable generation: {end - start} seconds')

matched_files = []
start = time.time()
for file in files_red_laser:
    if (file.number_of_molecules>4):
        print(file)
        file.sequencing_match = None
        # file.find_sequences3(distance=15, alpha=0.8, sigma=5, K_threshold=10**7)
        file.find_sequences3(distance=10, alpha=0.9, sigma=5, K_threshold=10**9, channel=1)
        if file.sequencing_match:
            matched_files.append(file)
end = time.time()
print(f'Matching: {end - start} seconds')

for file in matched_files:
    print(file.name + ' __ ' + str(file.sequencing_match.tile))

# Improve mapping by performing a linear least-squares fit on all nearest neighbours within the distance threshold
for file in matched_files:
    print(file)
    # try:
    nearest_neighbour_match_distance_threshold = 25
    file.sequencing_match.nearest_neighbour_match(nearest_neighbour_match_distance_threshold, transformation_type='linear')
    # except:
    #     print(file)

exp.show_sequencing_matches()

for file in matched_files:
    file.plot_sequencing_match()
    file.export_sequencing_match()


# -----------------------------------
# Find sequencing matches using stage coordinates
# -----------------------------------
# Find mapping between stage coordinates and sequencing coordinates
exp.map_sequencing_and_stage_coordinates()
exp.show_stage_to_sequencing_mappings()

# Find sequencing matches using stage coordinates
for file in files_red_laser:
    file.find_sequences_using_stage_coordinates(channel=1, show=True, save=False)

exp.show_sequencing_matches()

# Improve mapping by performing a linear least-squares fit on all nearest neighbours within the distance threshold
for file in files_red_laser:
    print(file)
    file.sequencing_match.nearest_neighbour_match(25, 'linear')
    file.sequencing_match.nearest_neighbour_match(10, 'linear')
    # file.sequencing_match.nearest_neighbour_match(10, 'polynomial', order=4)

file_red.sequencing_match.show_mapping_transformation(inverse=True, crop=True)


# -----------------------------------
# Classify traces in sequencing data of file
# -----------------------------------
# Doing this on all sequencing data is still slow.

