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

size_thresholds = [18.3, 14.4, 14.8, 12.5, 14.600000000000001]

n_cells = []
DATA = []
for T, TIME in enumerate(TIMES):
    DATA.append([])
    n_cells.append([])
    print()
    print(TIME)
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS
    
    size_th = size_thresholds[T]

    for COND in CONDS:
        DATA[-1].append([])
        n_cells[-1].append([])
        print(COND)
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
            # CT.plot(plot_args)
            
            # labs_to_rem = []
            # for cell in CT.jitcells:
            #     zc = int(cell.centers[0][0])
            #     zcid = cell.zs[0].index(zc)

            #     mask = cell.masks[0][zcid]
            #     area = len(mask) / CT.metadata["XYresolution"]**2
            #     if area < size_th:
            #         labs_to_rem.append(cell.label)
                
            # for lab in labs_to_rem:
            #     CT._del_cell(lab)  

            # CT.update_labels()

            # # Remove cells at extremes and perform z-drift correction
            # labs_to_rem = []
            # z_min = -1
            # data = [[] for i in range(4)]
            # zs = []
            # for ch in range(CT.hyperstack.shape[2]):
            #     correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
            #     if ch==3:
            #         z_min = np.argmax(intensity_profile)
            #     stack = CT.hyperstack[0,:,ch].astype("float32")
            #     for z in range(stack.shape[0]):
            #         stack[z] = stack[z] / correction_function[z]
            #     stack *= np.mean(intensity_profile)
            #     CT.hyperstack[0,:,ch] = stack.astype("uint8")
                
            #     for cell in CT.jitcells:
            #         z = int(cell.centers[0][0])
            #         if z < z_min:
            #             labs_to_rem.append(cell.label)
            #         if z > (len(correction_function) - z_min):
            #             labs_to_rem.append(cell.label)
                        
            #     for lab in labs_to_rem:
            #         print(lab)
            #         CT._del_cell(lab)  
                    
            # CT.update_labels()

            DATA[-1][-1].append([])
            n_cells[-1][-1].append(len(CT.jitcells))
            for ch_name in channel_names:
                DATA[-1][-1][-1].append([])
                
            for cell in CT.jitcells:
                z = int(cell.centers[0][0])
                zid = cell.zs[0].index(z)
                center = cell.centers[0][1:]
                mask = cell.masks[0][zid]  
                for ch in range(CT.hyperstack.shape[2]):
                    img = CT.hyperstack[0, z, ch]
                    val = np.maximum(0, np.mean(img[mask[:, 1], mask[:, 0]]) - np.mean(img[0:50, 0:50]))
                    DATA[-1][-1][-1][ch].append(val)
                
NANOG = []
CDX2 = []
OTX2 = []
DAPI = []

for T, TIME in enumerate(TIMES):
    NANOG.append([])
    CDX2.append([])
    OTX2.append([])
    DAPI.append([])
    
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    for C, COND in enumerate(CONDS):
        NANOG[-1].append([])
        CDX2[-1].append([])
        OTX2[-1].append([])
        DAPI[-1].append([])
        
for T, TIME in enumerate(TIMES):
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    data_chs = [[] for ch in channel_names]
    for C, COND in enumerate(CONDS):
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)
        try: 
            files = get_file_names(path_save_dir)
        except: 
            import os
            os.mkdir(path_save_dir)
         
        ### GET FULL FILE NAME AND FILE CODE ###
        ch_count = 0
        files = get_file_names(path_data_dir)
        g = -1
        for f, file in enumerate(files):
            if not ".tif" in file: continue
            if file in files_exclude: continue
            
            g+=1
            for ch, ch_name in enumerate(channel_names):
                data = DATA[T][C][g][ch]
                print(data)
                if ch==0:
                    NANOG[T][C] = [*NANOG[T][C], *data]
                elif ch==1:
                    CDX2[T][C] = [*CDX2[T][C], *data]
                elif ch==2:
                    OTX2[T][C] = [*OTX2[T][C], *data]
                elif ch==3:
                    DAPI[T][C] = [*DAPI[T][C], *data]
                
                ch_count+=1


import numpy as np
import matplotlib.pyplot as plt

# Function to remove outliers using IQR method
def remove_outliers(data):
    # Calculate the first and third quartiles (Q1 and Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 3.5 * IQR
    upper_bound = Q3 + 3.5 * IQR

    # Filter data to remove outliers
    return [x for x in data if lower_bound <= x <= upper_bound]

# Function to remove outliers using IQR method
def remove_outliers_pairs(data1, data2):
    # Calculate the first and third quartiles (Q1 and Q3)
    Q11 = np.percentile(data1, 25)
    Q31 = np.percentile(data1, 75)
    IQR1 = Q31 - Q11

    # Define the lower and upper bounds for outliers
    lower_bound1 = Q11 - 3.5 * IQR1
    upper_bound1 = Q31 + 3.5 * IQR1

    # Calculate the first and third quartiles (Q1 and Q3)
    Q12 = np.percentile(data2, 25)
    Q32 = np.percentile(data2, 75)
    IQR2 = Q32 - Q12

    # Define the lower and upper bounds for outliers
    lower_bound2 = Q12 - 3.5 * IQR2
    upper_bound2 = Q32 + 3.5 * IQR2

    final_data1 = []
    final_data2 = []
    # Filter data to remove outliers
    for i in range(len(data1)):
        if lower_bound1 <= data1[i] <= upper_bound1:
            if lower_bound2 <= data2[i] <= upper_bound2:
                final_data1.append(data1[i])
                final_data2.append(data2[i])
    return final_data1, final_data2

import numpy as np

def remove_outliers_lists(data_lists, iqr_multiplier=3.5):
    """
    Removes rows (i.e., cell entries) where any channel has an outlier.
    
    Parameters:
        data_lists (list of lists): Each inner list is a channel of the same length.
        iqr_multiplier (float): IQR multiplier for outlier detection.
        
    Returns:
        list of lists: Filtered data with outlier rows removed, preserving structure.
    """
    arrays = [np.array(lst) for lst in data_lists]
    n = len(arrays[0])

    # Sanity check
    if not all(len(arr) == n for arr in arrays):
        raise ValueError("All channel lists must be the same length.")

    keep_mask = np.ones(n, dtype=bool)

    for arr in arrays:
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        keep_mask &= (arr >= lower) & (arr <= upper)

    # Filter all arrays using the common mask
    filtered = [arr[keep_mask].tolist() for arr in arrays]
    return filtered

# Apply the IQR method to clean each dataset
NANOG_cleaned_48h_0 = remove_outliers(np.array(NANOG[0][0]))
NANOG_cleaned_48h_1 = remove_outliers(np.array(NANOG[0][1]))

CDX2_cleaned_48h_0 = remove_outliers(np.array(CDX2[0][0]))
CDX2_cleaned_48h_1 = remove_outliers(np.array(CDX2[0][1]))

OTX2_cleaned_48h_0 = remove_outliers(np.array(OTX2[0][0]))
OTX2_cleaned_48h_1 = remove_outliers(np.array(OTX2[0][1]))

DAPI_cleaned_48h_0 = remove_outliers(np.array(DAPI[0][0]))
DAPI_cleaned_48h_1 = remove_outliers(np.array(DAPI[0][1]))

# Repeat for other time points (60h, 72h)
NANOG_cleaned_60h_0 = remove_outliers(np.array(NANOG[1][0]))
NANOG_cleaned_60h_1 = remove_outliers(np.array(NANOG[1][1]))
NANOG_cleaned_60h_2 = remove_outliers(np.array(NANOG[1][2]))

CDX2_cleaned_60h_0 = remove_outliers(np.array(CDX2[1][0]))
CDX2_cleaned_60h_1 = remove_outliers(np.array(CDX2[1][1]))
CDX2_cleaned_60h_2 = remove_outliers(np.array(CDX2[1][2]))

OTX2_cleaned_60h_0 = remove_outliers(np.array(OTX2[1][0]))
OTX2_cleaned_60h_1 = remove_outliers(np.array(OTX2[1][1]))
OTX2_cleaned_60h_2 = remove_outliers(np.array(OTX2[1][2]))

DAPI_cleaned_60h_0 = remove_outliers(np.array(DAPI[1][0]))
DAPI_cleaned_60h_1 = remove_outliers(np.array(DAPI[1][1]))
DAPI_cleaned_60h_2 = remove_outliers(np.array(DAPI[1][2]))

NANOG_cleaned_72h_0 = remove_outliers(np.array(NANOG[2][0]))
NANOG_cleaned_72h_1 = remove_outliers(np.array(NANOG[2][1]))
NANOG_cleaned_72h_2 = remove_outliers(np.array(NANOG[2][2]))

CDX2_cleaned_72h_0 = remove_outliers(np.array(CDX2[2][0]))
CDX2_cleaned_72h_1 = remove_outliers(np.array(CDX2[2][1]))
CDX2_cleaned_72h_2 = remove_outliers(np.array(CDX2[2][2]))

OTX2_cleaned_72h_0 = remove_outliers(np.array(OTX2[2][0]))
OTX2_cleaned_72h_1 = remove_outliers(np.array(OTX2[2][1]))
OTX2_cleaned_72h_2 = remove_outliers(np.array(OTX2[2][2]))

DAPI_cleaned_72h_0 = remove_outliers(np.array(DAPI[2][0]))
DAPI_cleaned_72h_1 = remove_outliers(np.array(DAPI[2][1]))
DAPI_cleaned_72h_2 = remove_outliers(np.array(DAPI[2][2]))

NANOG_cleaned_84h_0 = remove_outliers(np.array(NANOG[3][0]))
NANOG_cleaned_84h_1 = remove_outliers(np.array(NANOG[3][1]))
NANOG_cleaned_84h_2 = remove_outliers(np.array(NANOG[3][2]))

CDX2_cleaned_84h_0 = remove_outliers(np.array(CDX2[3][0]))
CDX2_cleaned_84h_1 = remove_outliers(np.array(CDX2[3][1]))
CDX2_cleaned_84h_2 = remove_outliers(np.array(CDX2[3][2]))

OTX2_cleaned_84h_0 = remove_outliers(np.array(OTX2[3][0]))
OTX2_cleaned_84h_1 = remove_outliers(np.array(OTX2[3][1]))
OTX2_cleaned_84h_2 = remove_outliers(np.array(OTX2[3][2]))

DAPI_cleaned_84h_0 = remove_outliers(np.array(DAPI[3][0]))
DAPI_cleaned_84h_1 = remove_outliers(np.array(DAPI[3][1]))
DAPI_cleaned_84h_2 = remove_outliers(np.array(DAPI[3][2]))

NANOG_cleaned_96h_0 = remove_outliers(np.array(NANOG[4][0]))
NANOG_cleaned_96h_1 = remove_outliers(np.array(NANOG[4][1]))
NANOG_cleaned_96h_2 = remove_outliers(np.array(NANOG[4][2]))

CDX2_cleaned_96h_0 = remove_outliers(np.array(CDX2[4][0]))
CDX2_cleaned_96h_1 = remove_outliers(np.array(CDX2[4][1]))
CDX2_cleaned_96h_2 = remove_outliers(np.array(CDX2[4][2]))

OTX2_cleaned_96h_0 = remove_outliers(np.array(OTX2[4][0]))
OTX2_cleaned_96h_1 = remove_outliers(np.array(OTX2[4][1]))
OTX2_cleaned_96h_2 = remove_outliers(np.array(OTX2[4][2]))

DAPI_cleaned_96h_0 = remove_outliers(np.array(DAPI[4][0]))
DAPI_cleaned_96h_1 = remove_outliers(np.array(DAPI[4][1]))
DAPI_cleaned_96h_2 = remove_outliers(np.array(DAPI[4][2]))

bins = 75
fig, ax = plt.subplots(4, 5, figsize=(16, 10), sharex='row')

# 48 hours
ax[0, 0].hist(NANOG_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS_48[0], density=True)
ax[0, 0].hist(NANOG_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, label=CONDITIONS_48[1], density=True)
ax[0, 0].set_xlabel("NANOG")
ax[0, 0].legend(loc="upper right")

ax[1, 0].hist(CDX2_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 0].hist(CDX2_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, density=True)
ax[1, 0].set_xlabel("CDX2")

ax[2, 0].hist(OTX2_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 0].hist(OTX2_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, density=True)
ax[2, 0].set_xlabel("OTX2")

ax[3, 0].hist(DAPI_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 0].hist(DAPI_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, density=True)
ax[3, 0].set_xlabel("DAPI")

# 60 hours
ax[0, 1].hist(NANOG_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0], density=True)
ax[0, 1].hist(NANOG_cleaned_60h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1], density=True)
ax[0, 1].hist(NANOG_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2], density=True)
ax[0, 1].set_xlabel("NANOG")
ax[0, 1].legend(loc="upper right")

ax[1, 1].hist(CDX2_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 1].hist(CDX2_cleaned_60h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[1, 1].hist(CDX2_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[1, 1].set_xlabel("CDX2")

ax[2, 1].hist(OTX2_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 1].hist(OTX2_cleaned_60h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[2, 1].hist(OTX2_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[2, 1].set_xlabel("OTX2")

ax[3, 1].hist(DAPI_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 1].hist(DAPI_cleaned_60h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[3, 1].hist(DAPI_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[3, 1].set_xlabel("DAPI")

# 72 hours
ax[0, 2].hist(NANOG_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0], density=True)
ax[0, 2].hist(NANOG_cleaned_72h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1], density=True)
ax[0, 2].hist(NANOG_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2], density=True)
ax[0, 2].set_xlabel("NANOG")
ax[0, 2].legend(loc="upper right")

ax[1, 2].hist(CDX2_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 2].hist(CDX2_cleaned_72h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[1, 2].hist(CDX2_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[1, 2].set_xlabel("CDX2")

ax[2, 2].hist(OTX2_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 2].hist(OTX2_cleaned_72h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[2, 2].hist(OTX2_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[2, 2].set_xlabel("OTX2")

ax[3, 2].hist(DAPI_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 2].hist(DAPI_cleaned_72h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[3, 2].hist(DAPI_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[3, 2].set_xlabel("DAPI")

# 84 hours
ax[0, 3].hist(NANOG_cleaned_84h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0], density=True)
ax[0, 3].hist(NANOG_cleaned_84h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1], density=True)
ax[0, 3].hist(NANOG_cleaned_84h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2], density=True)
ax[0, 3].set_xlabel("NANOG")
ax[0, 3].legend(loc="upper right")

ax[1, 3].hist(CDX2_cleaned_84h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 3].hist(CDX2_cleaned_84h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[1, 3].hist(CDX2_cleaned_84h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[1, 3].set_xlabel("CDX2")

ax[2, 3].hist(OTX2_cleaned_84h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 3].hist(OTX2_cleaned_84h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[2, 3].hist(OTX2_cleaned_84h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[2, 3].set_xlabel("OTX2")

ax[3, 3].hist(DAPI_cleaned_84h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 3].hist(DAPI_cleaned_84h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[3, 3].hist(DAPI_cleaned_84h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[3, 3].set_xlabel("DAPI")

# 96 hours
ax[0, 4].hist(NANOG_cleaned_96h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0], density=True)
ax[0, 4].hist(NANOG_cleaned_96h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1], density=True)
ax[0, 4].hist(NANOG_cleaned_96h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2], density=True)
ax[0, 4].set_xlabel("NANOG")
ax[0, 4].legend(loc="upper right")

ax[1, 4].hist(CDX2_cleaned_96h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 4].hist(CDX2_cleaned_96h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[1, 4].hist(CDX2_cleaned_96h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[1, 4].set_xlabel("CDX2")

ax[2, 4].hist(OTX2_cleaned_96h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 4].hist(OTX2_cleaned_96h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[2, 4].hist(OTX2_cleaned_96h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[2, 4].set_xlabel("OTX2")

ax[3, 4].hist(DAPI_cleaned_96h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 4].hist(DAPI_cleaned_96h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[3, 4].hist(DAPI_cleaned_96h_2, color="yellow", alpha=0.5, bins=bins, density=True)
ax[3, 4].set_xlabel("DAPI")

ax[0, 0].set_ylabel("density")
ax[1, 0].set_ylabel("density")
ax[2, 0].set_ylabel("density")
ax[3, 0].set_ylabel("density")

ax[0, 0].set_title("48 hours")
ax[0, 1].set_title("60 hours")
ax[0, 2].set_title("72 hours")

plt.tight_layout()
plt.savefig(path_save_figs+"quantification_nanog.pdf")
plt.savefig(path_save_figs+"quantification_nanog.svg")
plt.show()

bins = 75
fig, ax = plt.subplots(4, 3, figsize=(16, 10), sharex='row')

# 48 hours
ax[0, 0].hist(NANOG_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS_48[0], density=True)
ax[0, 0].hist(NANOG_cleaned_48h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS_48[1], density=True)
ax[0, 0].set_xlabel("NANOG")
ax[0, 0].legend(loc="upper right")

ax[1, 0].hist(CDX2_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 0].hist(CDX2_cleaned_48h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[1, 0].set_xlabel("CDX2")

ax[2, 0].hist(OTX2_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 0].hist(OTX2_cleaned_48h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[2, 0].set_xlabel("OTX2")

ax[3, 0].hist(DAPI_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 0].hist(DAPI_cleaned_48h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[3, 0].set_xlabel("DAPI")

# 60 hours
ax[0, 1].hist(NANOG_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0], density=True)
ax[0, 1].hist(NANOG_cleaned_60h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1], density=True)
ax[0, 1].set_xlabel("NANOG")
ax[0, 1].legend(loc="upper right")

ax[1, 1].hist(CDX2_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 1].hist(CDX2_cleaned_60h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[1, 1].set_xlabel("CDX2")

ax[2, 1].hist(OTX2_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 1].hist(OTX2_cleaned_60h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[2, 1].set_xlabel("OTX2")

ax[3, 1].hist(DAPI_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 1].hist(DAPI_cleaned_60h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[3, 1].set_xlabel("DAPI")

# 72 hours
ax[0, 2].hist(NANOG_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0], density=True)
ax[0, 2].hist(NANOG_cleaned_72h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1], density=True)
ax[0, 2].set_xlabel("NANOG")
ax[0, 2].legend(loc="upper right")

ax[1, 2].hist(CDX2_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[1, 2].hist(CDX2_cleaned_72h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[1, 2].set_xlabel("CDX2")

ax[2, 2].hist(OTX2_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[2, 2].hist(OTX2_cleaned_72h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[2, 2].set_xlabel("OTX2")

ax[3, 2].hist(DAPI_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, density=True)
ax[3, 2].hist(DAPI_cleaned_72h_1, color="green", alpha=0.5, bins=bins, density=True)
ax[3, 2].set_xlabel("DAPI")

ax[0, 0].set_ylabel("density")
ax[1, 0].set_ylabel("density")
ax[2, 0].set_ylabel("density")
ax[3, 0].set_ylabel("density")

ax[0, 0].set_title("48 hours")
ax[0, 1].set_title("60 hours")
ax[0, 2].set_title("72 hours")

plt.tight_layout()
plt.savefig(path_save_figs+"quantification_noDMSO_nanog.pdf")
plt.savefig(path_save_figs+"quantification_noDMSO_nanog.svg")
plt.show()

NANOG_test = []
CDX2_test = []
OTX2_test = []
DAPI_test = []

for T, TIME in enumerate(TIMES):
    NANOG_test.append([])
    CDX2_test.append([])
    OTX2_test.append([])
    DAPI_test.append([])
    
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    for C, COND in enumerate(CONDS):
        NANOG_test[-1].append([])
        CDX2_test[-1].append([])
        OTX2_test[-1].append([])
        DAPI_test[-1].append([])
        
for T, TIME in enumerate(TIMES):
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    data_chs = [[] for ch in channel_names]
    for C, COND in enumerate(CONDS):
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)
        try: 
            files = get_file_names(path_save_dir)
        except: 
            import os
            os.mkdir(path_save_dir)
         
        ### GET FULL FILE NAME AND FILE CODE ###
        ch_count = 0
        files = get_file_names(path_data_dir)
        g = -1
        for f, file in enumerate(files):
            if not ".tif" in file: continue
            if file in files_exclude: continue
            
            g+=1
            for ch, ch_name in enumerate(channel_names):
                data = DATA[T][C][g][ch]
                if ch==0:
                    NANOG_test[T][C].append(data)
                elif ch==1:
                    CDX2_test[T][C].append(data)
                elif ch==2:
                    OTX2_test[T][C].append(data)
                elif ch==3:
                    DAPI_test[T][C].append(data)
            
                ch_count+=1
                print(ch_count)


fig, ax = plt.subplots()
ax.set_title("CDX2 in individual gastruloids")
for g, data in enumerate(CDX2_test[2][1]):
    ax.hist(data, alpha=0.5, bins = 100, label = "gastruloid {}".format(g), density=True)
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title("NANOG in individual gastruloids")
for g, data in enumerate(NANOG_test[2][1]):
    ax.hist(data, alpha=0.5, bins = 100, label = "gastruloid {}".format(g), density=True)
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title("NANOG in individual gastruloids")
for g, data in enumerate(OTX2_test[2][1]):
    ax.hist(data, alpha=0.5, bins = 100, label = "gastruloid {}".format(g), density=True)
plt.legend()
plt.show()

bins = 30

for T, TIME in enumerate(TIMES):
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    for C, COND in enumerate(CONDS):
            
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)

        files = get_file_names(path_data_dir)
        g = 0
        for f, file in enumerate(files):
            if not ".tif" in file: continue
            if file in files_exclude: continue
            # tif_reader_5D(file)
            g+=1

        fig, ax = plt.subplots(g+1,6, figsize=((g+1)*2.5,8), sharex='col')
        ax_count = 0
        g = 0

        ch = 0 
        ch_name = channel_names[ch]
        plot_titles=True
        for f, file in enumerate(files):
            if not ".tif" in file: continue
            if file in files_exclude: continue
            hyperstack, metadata = tif_reader_5D(path_data_dir+file)
            z_mid = int(np.round(hyperstack.shape[1]/2))
            print(file)
            
            if plot_titles:
                ax[g, 0].set_title("NANOG")
                ax[g, 1].set_title("NANOG")
                ax[g, 2].set_title("CDX2")
                ax[g, 3].set_title("CDX2")
                ax[g, 4].set_title("OTX2")
                ax[g, 5].set_title("OTX2")
                plot_titles=False
                
            ax[g, 0].hist(remove_outliers(NANOG_test[T][C][g]), alpha=0.5, bins=bins, density=True)
            ax[-1, 0].hist(remove_outliers(NANOG_test[T][C][g]), alpha=0.5, bins=bins, density=True)
            ax[g, 1].imshow(hyperstack[0,z_mid,0], vmin=0, vmax=100)
            ax[g, 2].hist(remove_outliers(CDX2_test[T][C][g]), alpha=0.5, bins=bins, density=True)
            ax[-1, 2].hist(remove_outliers(CDX2_test[T][C][g]), alpha=0.5, bins=bins, density=True)
            ax[g, 3].imshow(hyperstack[0,z_mid,1], vmin=0, vmax=100)
            ax[g, 4].hist(remove_outliers(OTX2_test[T][C][g]), alpha=0.5, bins=bins, density=True)
            ax[-1, 4].hist(remove_outliers(OTX2_test[T][C][g]), alpha=0.5, bins=bins, density=True)
            ax[g, 5].imshow(hyperstack[0,z_mid,2], vmin=0, vmax=100)
            g+=1
        
        ax[-1, 1].axis('off')
        ax[-1, 3].axis('off')
        ax[-1, 5].axis('off')

        plt.tight_layout()
        plt.savefig(path_save_figs+TIME+"_"+COND+".svg")
        plt.savefig(path_save_figs+TIME+"_"+COND+".pdf")

plt.show()

nanog_th1 = np.percentile(NANOG[0][0], 99.5)
nanog_th2 = np.percentile(NANOG[0][1], 99.5)

nanog_th = np.mean([nanog_th1, nanog_th2])

cdx2_th1 = np.percentile(CDX2[0][0], 99.5)
cdx2_th2 = np.percentile(CDX2[0][1], 99.5)

cdx2_th = np.mean([cdx2_th1, cdx2_th2])

from scipy.stats import gaussian_kde
# for T, TIME in enumerate(TIMES[:3]):
#     if TIME=="48h":
#         CONDS = CONDITIONS_48
#     else:
#         CONDS = CONDITIONS

#     for C, COND in enumerate(CONDS):
                    
#         fig, ax = plt.subplots(2,3, figsize=(16,10))

#         fig.suptitle(TIME + " " + COND )
#         data1, data2 = remove_outliers_pairs(NANOG[T][C], CDX2[T][C])
#         # Calculate the point density
#         data12 = np.vstack([np.log(data1),np.log(data2)])
#         cols = gaussian_kde(data12)(data12)
#         ax[0,0].scatter(data1, data2, s=1, c=cols)
#         ax[0,0].set_xlabel("NANOG")
#         ax[0,0].set_ylabel("CDX2")

#         data1, data2 = remove_outliers_pairs(NANOG[T][C], DAPI[T][C])
#         # Calculate the point density
#         data12 = np.vstack([np.log(data1),np.log(data2)])
#         cols = gaussian_kde(data12)(data12)
#         ax[0,1].scatter(data1, data2, s=1, c=cols)
#         ax[0,1].set_xlabel("NANOG")
#         ax[0,1].set_ylabel("DAPI")

#         data1, data2 = remove_outliers_pairs(NANOG[T][C], OTX2[T][C])
#         # Calculate the point density
#         data12 = np.vstack([np.log(data1),np.log(data2)])
#         cols = gaussian_kde(data12)(data12)
#         ax[0,2].scatter(data1, data2, s=1, c=cols)
#         ax[0,2].set_xlabel("NANOG")
#         ax[0,2].set_ylabel("OTX2")

#         data1, data2 = remove_outliers_pairs(CDX2[T][C], DAPI[T][C])
#         # Calculate the point density
#         data12 = np.vstack([np.log(data1),np.log(data2)])
#         cols = gaussian_kde(data12)(data12)
#         ax[1,0].scatter(data1, data2, s=1, c=cols)
#         ax[1,0].set_xlabel("CDX2")
#         ax[1,0].set_ylabel("DAPI")

#         data1, data2 = remove_outliers_pairs(CDX2[T][C], OTX2[T][C])
#         # Calculate the point density
#         data12 = np.vstack([np.log(data1),np.log(data2)])
#         cols = gaussian_kde(data12)(data12)
#         ax[1,1].scatter(data1, data2, s=1, c=cols)
#         ax[1,1].set_xlabel("CDX2")
#         ax[1,1].set_ylabel("OTX2")

#         data1, data2 = remove_outliers_pairs(OTX2[T][C], DAPI[T][C])
#         # Calculate the point density
#         data12 = np.vstack([np.log(data1),np.log(data2)])
#         cols = gaussian_kde(data12)(data12)
#         ax[1,2].scatter(data1, data2, s=1, c=cols)
#         ax[1,2].set_xlabel("OTX2")
#         ax[1,2].set_ylabel("DAPI")

#         plt.tight_layout()
#         plt.savefig(path_save_figs+"scatters/"+TIME+"_"+COND+".pdf")

# plt.show()

fig, ax = plt.subplots(2,3, figsize=(15,10), sharex=True, sharey=True)

data1, data2 = remove_outliers_pairs(NANOG[0][1], CDX2[0][1])
# Calculate the point density
data12 = np.vstack([data1,data2])
cols = gaussian_kde(data12)(data12)
ax[0,0].scatter(data1, data2, s=1, c=cols)
ax[0,0].axhline(cdx2_th, c="k", lw=2)
ax[0,0].axvline(nanog_th, c="k", lw=2)
ax[0,0].set_ylabel("CDX2")

ax[0,1].set_title("WT CHIRON")
data1, data2 = remove_outliers_pairs(NANOG[1][1], CDX2[1][1])
# Calculate the point density
data12 = np.vstack([data1,data2])
cols = gaussian_kde(data12)(data12)
ax[0,1].scatter(data1, data2, s=0.5, c=cols)
ax[0,1].axhline(cdx2_th, c="k", lw=2)
ax[0,1].axvline(nanog_th, c="k", lw=2)

data1, data2 = remove_outliers_pairs(NANOG[2][1], CDX2[2][1])
# Calculate the point density
data12 = np.vstack([data1,data2])
cols = gaussian_kde(data12)(data12)
ax[0,2].scatter(data1, data2, s=0.5, c=cols)
ax[0,2].axhline(cdx2_th, c="k", lw=2)
ax[0,2].axvline(nanog_th, c="k", lw=2)

data1, data2 = remove_outliers_pairs(NANOG[0][0], CDX2[0][0])
# Calculate the point density
data12 = np.vstack([data1,data2])
cols = gaussian_kde(data12)(data12)
ax[1,0].scatter(data1, data2, s=0.5, c=cols)
ax[1,0].axhline(cdx2_th, c="k", lw=2)
ax[1,0].axvline(nanog_th, c="k", lw=2)
ax[1,0].set_xlabel("NANOG")
ax[1,0].set_ylabel("CDX2")

ax[1,1].set_title("Wnt3KO")
data1, data2 = remove_outliers_pairs(NANOG[1][0], CDX2[1][0])
# Calculate the point density
data12 = np.vstack([data1,data2])
cols = gaussian_kde(data12)(data12)
ax[1,1].scatter(data1, data2, s=0.5, c=cols)
ax[1,1].axhline(cdx2_th, c="k", lw=2)
ax[1,1].axvline(nanog_th, c="k", lw=2)
ax[1,1].set_xlabel("NANOG")

data1, data2 = remove_outliers_pairs(NANOG[2][0], CDX2[2][0])
# Calculate the point density
data12 = np.vstack([data1,data2])
cols = gaussian_kde(data12)(data12)
ax[1,2].scatter(data1, data2, s=0.5, c=cols)
ax[1,2].axhline(cdx2_th, c="k", lw=2)
ax[1,2].axvline(nanog_th, c="k", lw=2)
ax[1,2].set_xlabel("NANOG")

plt.show()

