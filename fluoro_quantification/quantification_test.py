### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, get_intenity_profile, get_file_names, construct_RGB, extract_fluoro, tif_reader_5D
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')
### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
EXP = "Nanog_Cdx2_Otx2_DAPI"
TIMES = ["48h", "60h", "72h"]
# TIMES = ["48h", "60h", "72h", "84h"]

CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]
channel_names = ["NANOG", "CDX2", "OTX2", "DAPI"]

files_exclude = [
    "G6-E14 48h SOX2 647 OCT4 546 BRA 488 DAPI_11.tif",
    "G4-WNT3KO 60H DMSO SOX2 647 OCT4 555 BRA 488 DAPI_22.tif",
    "G2-E14 96h CHIR SOX2 647 OCT4 546 BRA 488 DAPI_17.tif",
    "G5-72h E14 CHIR NANOG647 CDX2_555 OTX2_488 DAPI_26.tif",
    "G4-E14 DMSO 96H NANOG_647 CDX2_555 OTX2_488 DAPI_20.tif",
    "G1-WNT3KO DMSO 96H NANOG_647 CDX2_555 OTX2_488 DAPI_31.tif",
    "G6-96h WNT3KO DMSO NANOG647 CDX2_555 OTX2_488 DAPI_26.tif"
]

NANOG = []
CDX2 = []
OTX2 = []
DAPI = []

for T, TIME in enumerate(TIMES):
    NANOG.append([])
    CDX2.append([])
    OTX2.append([])
    DAPI.append([])

    print()
    print(TIME)
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS
    
    for COND in CONDS:
        NANOG[-1].append([])
        CDX2[-1].append([])
        OTX2[-1].append([])
        DAPI[-1].append([])        
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
            
            NANOG[-1][-1].append([])
            CDX2[-1][-1].append([])
            OTX2[-1][-1].append([])
            DAPI[-1][-1].append([])
            
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
                
            CT.update_labels()
                        
            # ch = channel_names.index("DAPI")
            # correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)

            # for ch in range(CT.hyperstack.shape[2]):
            #     _correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
            #     stack = CT.hyperstack[0,:,ch].astype("float32")
            #     for z in range(stack.shape[0]):
            #         stack[z] = stack[z] / correction_function[z]
            #     stack *= np.mean(intensity_profile)
            #     CT.hyperstack[0,:,ch] = stack.astype("uint8")

            for cell in CT.jitcells:
                z = int(cell.centers[0][0])
                zid = cell.zs[0].index(z)
                center = cell.centers[0][1:]
                mask = cell.masks[0][zid]  
                
                img = CT.hyperstack[0, z, 0]
                NANOG[-1][-1][-1].append(np.mean(img[mask[:, 1], mask[:, 0]]))

                img = CT.hyperstack[0, z, 1]
                CDX2[-1][-1][-1].append(np.mean(img[mask[:, 1], mask[:, 0]]))
                
                img = CT.hyperstack[0, z, 2]
                OTX2[-1][-1][-1].append(np.mean(img[mask[:, 1], mask[:, 0]]))
                
                img = CT.hyperstack[0, z, 3]
                DAPI[-1][-1][-1].append(np.mean(img[mask[:, 1], mask[:, 0]]))


bins = 75

fig, ax = plt.subplots(4,3, figsize=(16,10))

ax[0,0].hist(NANOG[0][0], color="magenta", alpha=0.5, bins=bins, label=CONDITIONS_48[0])
ax[0,0].hist(NANOG[0][1], color="orange", alpha=0.5, bins=bins, label=CONDITIONS_48[1])
ax[0,0].legend()

ax[1,0].hist(CDX2[0][0], color="magenta", alpha=0.5, bins=bins)
ax[1,0].hist(CDX2[0][1], color="orange", alpha=0.5, bins=bins)

ax[2,0].hist(OTX2[0][0], color="magenta", alpha=0.5, bins=bins)
ax[2,0].hist(OTX2[0][1], color="orange", alpha=0.5, bins=bins)

ax[3,0].hist(DAPI[0][0], color="magenta", alpha=0.5, bins=bins)
ax[3,0].hist(DAPI[0][1], color="orange", alpha=0.5, bins=bins)

ax[0,1].hist(NANOG[1][0], color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[0,1].hist(NANOG[1][1], color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[0,1].hist(NANOG[1][2], color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[0,1].legend()

ax[1,1].hist(CDX2[1][0], color="magenta", alpha=0.5, bins=bins)
ax[1,1].hist(CDX2[1][1], color="green", alpha=0.5, bins=bins)
ax[1,1].hist(CDX2[1][2], color="yellow", alpha=0.5, bins=bins)

ax[2,1].hist(OTX2[1][0], color="magenta", alpha=0.5, bins=bins)
ax[2,1].hist(OTX2[1][1], color="green", alpha=0.5, bins=bins)
ax[2,1].hist(OTX2[1][2], color="yellow", alpha=0.5, bins=bins)

ax[3,1].hist(DAPI[1][0], color="magenta", alpha=0.5, bins=bins)
ax[3,1].hist(DAPI[1][1], color="green", alpha=0.5, bins=bins)
ax[3,1].hist(DAPI[1][2], color="yellow", alpha=0.5, bins=bins)

ax[0,2].hist(NANOG[2][0], color="magenta", alpha=0.5, bins=bins)
ax[0,2].hist(NANOG[2][1], color="green", alpha=0.5, bins=bins)
ax[0,2].hist(NANOG[2][2], color="yellow", alpha=0.5, bins=bins)

ax[1,2].hist(CDX2[2][0], color="magenta", alpha=0.5, bins=bins)
ax[1,2].hist(CDX2[2][1], color="green", alpha=0.5, bins=bins)
ax[1,2].hist(CDX2[2][2], color="yellow", alpha=0.5, bins=bins)

ax[2,2].hist(OTX2[2][0], color="magenta", alpha=0.5, bins=bins)
ax[2,2].hist(OTX2[2][1], color="green", alpha=0.5, bins=bins)
ax[2,2].hist(OTX2[2][2], color="yellow", alpha=0.5, bins=bins)

ax[3,2].hist(DAPI[2][0], color="magenta", alpha=0.5, bins=bins)
ax[3,2].hist(DAPI[2][1], color="green", alpha=0.5, bins=bins)
ax[3,2].hist(DAPI[2][2], color="yellow", alpha=0.5, bins=bins)

ax[0,0].set_ylabel("NANOG")
ax[1,0].set_ylabel("CDX2")
ax[2,0].set_ylabel("OTX2")
ax[3,0].set_ylabel("DAPI")

ax[0,0].set_title("48 hours")
ax[0,1].set_title("60 hours")
ax[0,2].set_title("72 hours")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Function to remove outliers using IQR method
def remove_outliers(data):
    # Calculate the first and third quartiles (Q1 and Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    # Filter data to remove outliers
    return [x for x in data if lower_bound <= x <= upper_bound]

# Apply the IQR method to clean each dataset
NANOG_cleaned_48h_0 = remove_outliers(NANOG[0][0])
NANOG_cleaned_48h_1 = remove_outliers(NANOG[0][1])

CDX2_cleaned_48h_0 = remove_outliers(CDX2[0][0])
CDX2_cleaned_48h_1 = remove_outliers(CDX2[0][1])

OTX2_cleaned_48h_0 = remove_outliers(OTX2[0][0])
OTX2_cleaned_48h_1 = remove_outliers(OTX2[0][1])

DAPI_cleaned_48h_0 = remove_outliers(DAPI[0][0])
DAPI_cleaned_48h_1 = remove_outliers(DAPI[0][1])

# Repeat for other time points (60h, 72h)
NANOG_cleaned_60h_0 = remove_outliers(NANOG[1][0])
NANOG_cleaned_60h_1 = remove_outliers(NANOG[1][1])
NANOG_cleaned_60h_2 = remove_outliers(NANOG[1][2])

CDX2_cleaned_60h_0 = remove_outliers(CDX2[1][0])
CDX2_cleaned_60h_1 = remove_outliers(CDX2[1][1])
CDX2_cleaned_60h_2 = remove_outliers(CDX2[1][2])

OTX2_cleaned_60h_0 = remove_outliers(OTX2[1][0])
OTX2_cleaned_60h_1 = remove_outliers(OTX2[1][1])
OTX2_cleaned_60h_2 = remove_outliers(OTX2[1][2])

DAPI_cleaned_60h_0 = remove_outliers(DAPI[1][0])
DAPI_cleaned_60h_1 = remove_outliers(DAPI[1][1])
DAPI_cleaned_60h_2 = remove_outliers(DAPI[1][2])

NANOG_cleaned_72h_0 = remove_outliers(NANOG[2][0])
NANOG_cleaned_72h_1 = remove_outliers(NANOG[2][1])
NANOG_cleaned_72h_2 = remove_outliers(NANOG[2][2])

CDX2_cleaned_72h_0 = remove_outliers(CDX2[2][0])
CDX2_cleaned_72h_1 = remove_outliers(CDX2[2][1])
CDX2_cleaned_72h_2 = remove_outliers(CDX2[2][2])

OTX2_cleaned_72h_0 = remove_outliers(OTX2[2][0])
OTX2_cleaned_72h_1 = remove_outliers(OTX2[2][1])
OTX2_cleaned_72h_2 = remove_outliers(OTX2[2][2])

DAPI_cleaned_72h_0 = remove_outliers(DAPI[2][0])
DAPI_cleaned_72h_1 = remove_outliers(DAPI[2][1])
DAPI_cleaned_72h_2 = remove_outliers(DAPI[2][2])

# Plot the cleaned data
bins = 75
fig, ax = plt.subplots(4, 3, figsize=(16, 10), sharex='row')

# 48 hours
ax[0, 0].hist(NANOG_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS_48[0])
ax[0, 0].hist(NANOG_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, label=CONDITIONS_48[1])
ax[0, 0].legend(loc="upper right")

ax[1, 0].hist(CDX2_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS_48[0])
ax[1, 0].hist(CDX2_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, label=CONDITIONS_48[1])
ax[1, 0].legend(loc="upper right")

ax[2, 0].hist(OTX2_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS_48[0])
ax[2, 0].hist(OTX2_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, label=CONDITIONS_48[1])
ax[2, 0].legend(loc="upper right")

ax[3, 0].hist(DAPI_cleaned_48h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS_48[0])
ax[3, 0].hist(DAPI_cleaned_48h_1, color="orange", alpha=0.5, bins=bins, label=CONDITIONS_48[1])
ax[3, 0].legend(loc="upper right")

# 60 hours
ax[0, 1].hist(NANOG_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[0, 1].hist(NANOG_cleaned_60h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[0, 1].hist(NANOG_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[0, 1].legend(loc="upper right")

ax[1, 1].hist(CDX2_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[1, 1].hist(CDX2_cleaned_60h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[1, 1].hist(CDX2_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[1, 1].legend(loc="upper right")

ax[2, 1].hist(OTX2_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[2, 1].hist(OTX2_cleaned_60h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[2, 1].hist(OTX2_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[2, 1].legend(loc="upper right")

ax[3, 1].hist(DAPI_cleaned_60h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[3, 1].hist(DAPI_cleaned_60h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[3, 1].hist(DAPI_cleaned_60h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[3, 1].legend(loc="upper right")

# 72 hours
ax[0, 2].hist(NANOG_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[0, 2].hist(NANOG_cleaned_72h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[0, 2].hist(NANOG_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[0, 2].legend(loc="upper right")

ax[1, 2].hist(CDX2_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[1, 2].hist(CDX2_cleaned_72h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[1, 2].hist(CDX2_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[1, 2].legend(loc="upper right")

ax[2, 2].hist(OTX2_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[2, 2].hist(OTX2_cleaned_72h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[2, 2].hist(OTX2_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[2, 2].legend(loc="upper right")

ax[3, 2].hist(DAPI_cleaned_72h_0, color="magenta", alpha=0.5, bins=bins, label=CONDITIONS[0])
ax[3, 2].hist(DAPI_cleaned_72h_1, color="green", alpha=0.5, bins=bins, label=CONDITIONS[1])
ax[3, 2].hist(DAPI_cleaned_72h_2, color="yellow", alpha=0.5, bins=bins, label=CONDITIONS[2])
ax[3, 2].legend(loc="upper right")

# Set y-axis labels and titles
ax[0, 0].set_ylabel("NANOG")
ax[1, 0].set_ylabel("CDX2")
ax[2, 0].set_ylabel("OTX2")
ax[3, 0].set_ylabel("DAPI")

ax[0, 0].set_title("48 hours")
ax[0, 1].set_title("60 hours")
ax[0, 2].set_title("72 hours")

plt.tight_layout()
plt.show()

bins = 30

T = 1
C = 1
TIME = TIMES[T]
COND = CONDITIONS[C]
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)

files = get_file_names(path_data_dir)
g = 0
for f, file in enumerate(files):
    if not ".tif" in file: continue
    if file in files_exclude: continue
    # tif_reader_5D(file)
    g+=1

fig, ax = plt.subplots(g,6, figsize=(g*4,6), sharex='col')
ax_count = 0
g = 0

fig.suptitle(TIME + " " + COND)
for f, file in enumerate(files):
    if not ".tif" in file: continue
    if file in files_exclude: continue
    hyperstack, metadata = tif_reader_5D(path_data_dir+file)
    z_mid = int(np.round(hyperstack.shape[1]/2))
    print(file)
    ax[g, 0].hist(NANOG[g], alpha=0.5, bins=bins)
    ax[g, 1].imshow(hyperstack[0,z_mid,0], vmin=0, vmax=100)
    ax[g, 2].hist(CDX2[g], alpha=0.5, bins=bins)
    ax[g, 3].imshow(hyperstack[0,z_mid,1], vmin=0, vmax=100)
    ax[g, 4].hist(OTX2[g], alpha=0.5, bins=bins)
    ax[g, 5].imshow(hyperstack[0,z_mid,2], vmin=0, vmax=100)
    g+=1
    
plt.tight_layout()
plt.show()

