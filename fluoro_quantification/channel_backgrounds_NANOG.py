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

CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR"]
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

size_thresholds = [18.3, 14.4, 14.8, 12.5, 14.600000000000001]

DATA = []
for T, TIME in enumerate(TIMES):
    DATA.append([])
    print()
    print(TIME)
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS
    
    size_th = size_thresholds[T]

    for COND in CONDS:
        DATA[-1].append([])
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
            labs_to_rem = []
            for cell in CT.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                mask = cell.masks[0][zcid]
                area = len(mask) / CT.metadata["XYresolution"]**2
                if area < size_th:
                    labs_to_rem.append(cell.label)
                
            for lab in labs_to_rem:
                CT._del_cell(lab)  
                
            CT.update_labels()
                        
            ch = channel_names.index("DAPI")
            correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)

            for ch in range(CT.hyperstack.shape[2]):
                _correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
                stack = CT.hyperstack[0,:,ch].astype("float32")
                for z in range(stack.shape[0]):
                    stack[z] = stack[z] / correction_function[z]
                stack *= np.mean(intensity_profile)
                CT.hyperstack[0,:,ch] = stack.astype("uint8")


            results = extract_fluoro(CT)

            for ch_name in channel_names:
                ch = channel_names.index(ch_name)
                vals = results["channel_{}".format(ch)]
                DATA[-1][-1].append(vals)
                
        
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
        for f, file in enumerate(files):
            if not ".tif" in file: continue
            for ch, ch_name in enumerate(channel_names):
                data = DATA[T][C][ch_count]
                if ch==0:
                    NANOG[T][C] = [*NANOG[T][C], *data]
                elif ch==1:
                    CDX2[T][C] = [*CDX2[T][C], *data]
                elif ch==2:
                    OTX2[T][C] = [*OTX2[T][C], *data]
                elif ch==3:
                    DAPI[T][C] = [*DAPI[T][C], *data]
                
                ch_count+=1

bins = 100

fig, ax = plt.subplots(4,5, figsize=(16,10))

ax[0,0].hist(NANOG[0][0], color="magenta", alpha=0.5, bins=bins)
ax[0,0].hist(NANOG[0][1], color="orange", alpha=0.5, bins=bins)

ax[1,0].hist(CDX2[0][0], color="magenta", alpha=0.5, bins=bins)
ax[1,0].hist(CDX2[0][1], color="orange", alpha=0.5, bins=bins)

ax[2,0].hist(OTX2[0][0], color="magenta", alpha=0.5, bins=bins)
ax[2,0].hist(OTX2[0][1], color="orange", alpha=0.5, bins=bins)

ax[3,0].hist(DAPI[0][0], color="magenta", alpha=0.5, bins=bins)
ax[3,0].hist(DAPI[0][1], color="orange", alpha=0.5, bins=bins)

ax[0,1].hist(NANOG[1][0], color="magenta", alpha=0.5, bins=bins)
ax[0,1].hist(NANOG[1][1], color="green", alpha=0.5, bins=bins)
ax[0,1].hist(NANOG[1][2], color="yellow", alpha=0.5, bins=bins)

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

ax[0,3].hist(NANOG[3][0], color="magenta", alpha=0.5, bins=bins)
ax[0,3].hist(NANOG[3][1], color="green", alpha=0.5, bins=bins)
ax[0,3].hist(NANOG[3][2], color="yellow", alpha=0.5, bins=bins)

ax[1,3].hist(CDX2[3][0], color="magenta", alpha=0.5, bins=bins)
ax[1,3].hist(CDX2[3][1], color="green", alpha=0.5, bins=bins)
ax[1,3].hist(CDX2[3][2], color="yellow", alpha=0.5, bins=bins)

ax[2,3].hist(OTX2[3][0], color="magenta", alpha=0.5, bins=bins)
ax[2,3].hist(OTX2[3][1], color="green", alpha=0.5, bins=bins)
ax[2,3].hist(OTX2[3][2], color="yellow", alpha=0.5, bins=bins)

ax[3,3].hist(DAPI[3][0], color="magenta", alpha=0.5, bins=bins)
ax[3,3].hist(DAPI[3][1], color="green", alpha=0.5, bins=bins)
ax[3,3].hist(DAPI[3][2], color="yellow", alpha=0.5, bins=bins)

ax[0,4].hist(NANOG[4][0], color="magenta", alpha=0.5, bins=bins)
ax[0,4].hist(NANOG[4][1], color="green", alpha=0.5, bins=bins)
ax[0,4].hist(NANOG[4][2], color="yellow", alpha=0.5, bins=bins)

ax[1,4].hist(CDX2[4][0], color="magenta", alpha=0.5, bins=bins)
ax[1,4].hist(CDX2[4][1], color="green", alpha=0.5, bins=bins)
ax[1,4].hist(CDX2[4][2], color="yellow", alpha=0.5, bins=bins)

ax[2,4].hist(OTX2[4][0], color="magenta", alpha=0.5, bins=bins)
ax[2,4].hist(OTX2[4][1], color="green", alpha=0.5, bins=bins)
ax[2,4].hist(OTX2[4][2], color="yellow", alpha=0.5, bins=bins)

ax[3,4].hist(DAPI[4][0], color="magenta", alpha=0.5, bins=bins)
ax[3,4].hist(DAPI[4][1], color="green", alpha=0.5, bins=bins)
ax[3,4].hist(DAPI[4][2], color="yellow", alpha=0.5, bins=bins)

ax[0,0].set_ylabel("NANOG")
ax[1,0].set_ylabel("CDX2")
ax[2,0].set_ylabel("OTX2")
ax[3,0].set_ylabel("DAPI")

ax[0,0].set_title("48 hours")
ax[0,1].set_title("60 hours")
ax[0,2].set_title("72 hours")
ax[0,3].set_title("84 hours")
ax[0,4].set_title("96 hours")

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
        for f, file in enumerate(files):
            if not ".tif" in file: continue
            for ch, ch_name in enumerate(channel_names):
                data = DATA[T][C][ch_count]
                if ch==0:
                    NANOG_test[T][C].append(data)
                elif ch==1:
                    CDX2_test[T][C].append(data)
                elif ch==2:
                    OTX2_test[T][C].append(data)
                elif ch==3:
                    DAPI_test[T][C].append(data)
            
                ch_count+=1

fig, ax = plt.subplots()
for data in CDX2_test[2][1]:
    ax.hist(data, alpha=0.5, bins = 100
)
plt.show()

T = 2
g = 2
fig, ax = plt.subplots(1,2)
ax[0].set_title("Wnt3KO")
ax[0].scatter(NANOG_test[T][0][g], CDX2_test[T][0][g])
ax[0].set_xlabel("NANOG")
ax[0].set_ylabel("CDX2")

ax[1].set_title("CHIRON")
ax[1].scatter(NANOG_test[T][1][g], CDX2_test[T][1][g])
ax[1].set_xlabel("NANOG")
ax[1].set_ylabel("CDX2")
plt.show()