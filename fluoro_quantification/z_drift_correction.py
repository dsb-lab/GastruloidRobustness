### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, get_intenity_profile, get_file_names, construct_RGB, extract_fluoro, tif_reader_5D
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_save_figs = '/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/Nanog_Cdx2_Otx2_DAPI/'

EXP = "Nanog_Cdx2_Otx2_DAPI"
TIMES = ["48h", "60h", "72h"]
# TIMES = ["48h", "60h", "72h", "84h", "96h"]

CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]
channel_names = ["NANOG", "CDX2", "OTX2", "DAPI"]


files_exclude = [
    "G6-E14 48h SOX2 647 OCT4 546 BRA 488 DAPI_11.tif",
    # "G4-WNT3KO 60H DMSO SOX2 647 OCT4 555 BRA 488 DAPI_22.tif",
    "G4-E14 48H NANOG_647 CDX2_555 OTX2_488 DAPI_31.tif",  # From z drift correction
    "G2-E14 96h CHIR SOX2 647 OCT4 546 BRA 488 DAPI_17.tif",
    "G5-72h E14 CHIR NANOG647 CDX2_555 OTX2_488 DAPI_26.tif",
    "G4-E14 DMSO 96H NANOG_647 CDX2_555 OTX2_488 DAPI_20.tif",
    "G1-WNT3KO DMSO 96H NANOG_647 CDX2_555 OTX2_488 DAPI_31.tif",
    "G6-96h WNT3KO DMSO NANOG647 CDX2_555 OTX2_488 DAPI_26.tif",
    "G2-E14 60H DMSO SOX2 647 OCT4 555 BRA 488 DAPI_22.tif",
    "G2-60h E14 DMSO NANOG647 CDX2_555 OTX2_488 DAPI-MOVED_18.tif" # From z drift correction
]


chs_prof = [[] for i in range(4)]
chs_corr = [[] for i in range(4)]
file_lab = []

TIME = TIMES[2]
print()
print(TIME)
if TIME=="48h":
    CONDS = CONDITIONS_48
else:
    CONDS = CONDITIONS
        
COND = CONDS[2]

path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)
try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
for file in files:
    if not ".tif" in file: continue
    if file in files_exclude: continue
    
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

    file_lab.append(file)
    for ch in range(CT.hyperstack.shape[2]):
        correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
        chs_prof[ch].append(intensity_profile)
        chs_corr[ch].append(correction_function)


import numpy as np

fig, ax = plt.subplots(2,2, sharex=True)

ch = 0
for l, line in enumerate(chs_corr[ch]):
    ax[0,0].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[0,0].set_ylabel(channel_names[ch])
ax[0,0].legend()

ch = 1
for l, line in enumerate(chs_corr[ch]):
    ax[0,1].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[0,1].set_ylabel(channel_names[ch])
ax[0,1].legend()

ch = 2
for l, line in enumerate(chs_corr[ch]):
    ax[1,0].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[1,0].set_ylabel(channel_names[ch])
    ax[1,0].set_xlabel("% of total depth")
ax[1,0].legend()

ch = 3
for l, line in enumerate(chs_corr[ch]):
    ax[1,1].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[1,1].set_ylabel(channel_names[ch])
    ax[1,1].set_xlabel("% of total depth")
ax[1,1].legend()

plt.show()


fig, ax = plt.subplots(2,2, sharex=True)

ch = 0
for l, line in enumerate(chs_prof[ch]):
    ax[0,0].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[0,0].set_ylabel(channel_names[ch])
ax[0,0].legend()

ch = 1
for l, line in enumerate(chs_prof[ch]):
    ax[0,1].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[0,1].set_ylabel(channel_names[ch])
ax[0,1].legend()

ch = 2
for l, line in enumerate(chs_prof[ch]):
    ax[1,0].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[1,0].set_ylabel(channel_names[ch])
    ax[1,0].set_xlabel("% of total depth")
ax[1,0].legend()

ch = 3
for l, line in enumerate(chs_prof[ch]):
    ax[1,1].plot(np.linspace(0, 100, len(line)), line, label=file_lab[l])
    ax[1,1].set_ylabel(channel_names[ch])
    ax[1,1].set_xlabel("% of total depth")
ax[1,1].legend()

plt.show()
