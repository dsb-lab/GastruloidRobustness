### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')
### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
EXPERIMENTS = ["Sox2_Oct4_Bra_DAPI", "Nanog_Cdx2_Otx2_DAPI"]
CHANNEL_NAMES = [["SOX2", "OCT4", "BRA", "DAPI"], ["NANOG", "CDX2", "OTX2", "DAPI"]]
TIMES = ["48h", "60h", "72h", "84h", "96h"]
CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]

exp=0
EXP = EXPERIMENTS[exp]
TIME = TIMES[-1]
if TIME=="48h":
    COND = CONDITIONS_48[1]
else:
    COND = CONDITIONS[1]

path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)
try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
files = [file for file in files if ".tif" in file]
file=files[0]

channel_names = CHANNEL_NAMES[exp]

file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)

path_data = path_data_dir+file
path_save = path_save_dir+embcode
try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': None, 
}

concatenation3D_args = {
    'distance_th_z': 5.0, # microns
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.1, 
    'min_cell_planes': 2,
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

ch = channel_names.index("DAPI")
chans = [ch]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

# Plot all channels except DAPI
chans_plot = [_ch for _ch in chans if _ch != ch]

batch_args = {
    'name_format':"ch"+str(ch)+"_{}",
    'extension':".tif",
}
plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch],
    # 'channels': chans_plot,
    'min_outline_length':75,
}

CT = cellSegTrack(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)

CT.load()
CT.plot(plot_args)
