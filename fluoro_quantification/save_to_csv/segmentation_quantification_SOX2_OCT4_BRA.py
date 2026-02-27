### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, check_or_create_dir, get_intenity_profile, get_file_names, construct_RGB, extract_fluoro, tif_reader_5D
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_save_figs="/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/Sox2_Oct4_Bra_DAPI/"

EXP = "Sox2_Oct4_Bra_DAPI"
TIMES = ["48h", "60h", "72h", "84h", "96h"]

CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]
channel_names = ["SOX2", "OCT4", "BRA", "DAPI"]

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

            # Remove cells at extremes and perform z-drift correction
            labs_to_rem = []
            z_min = -1
            data = [[] for i in range(4)]
            zs = []
            for ch in range(CT.hyperstack.shape[2]):
                correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
                if ch==3:
                    z_min = z_positions[np.argmax(intensity_profile)]
                stack = CT.hyperstack[0,:,ch].astype("float32")
                for z in range(stack.shape[0]):
                    stack[z] = stack[z] / correction_function[z]
                stack *= np.mean(intensity_profile)
                CT.hyperstack[0,:,ch] = stack.astype("uint8")
                
                for cell in CT.jitcells:
                    z = int(cell.centers[0][0])
                    if z < z_min:
                        labs_to_rem.append(cell.label)
                    if z > (len(correction_function) - z_min):
                        labs_to_rem.append(cell.label)
                                
            for lab in labs_to_rem:
                print(lab)
                CT._del_cell(lab)  
                    
            CT.update_labels()

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

path_results = "/home/pablo/Desktop/PhD/projects/GastruloidRobustness/results/{}/".format(EXP)
check_or_create_dir(path_results)
import pandas as pd

SOX2 = []
OCT4 = []
BRA = []
DAPI = []

for T, TIME in enumerate(TIMES):
    path_results_t = path_results+"{}/".format(TIME)
    check_or_create_dir(path_results_t)
    SOX2.append([])
    OCT4.append([])
    BRA.append([])
    DAPI.append([])
    
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    for C, COND in enumerate(CONDS):
        path_results_c = path_results_t+"{}/".format(COND)
        check_or_create_dir(path_results_c)
        SOX2[-1].append([])
        OCT4[-1].append([])
        BRA[-1].append([])
        DAPI[-1].append([])
        
for T, TIME in enumerate(TIMES):
    path_results_t = path_results+"{}/".format(TIME)
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    data_chs = [[] for ch in channel_names]
    for C, COND in enumerate(CONDS):
        path_results_c = path_results_t+"{}/".format(COND)
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
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            g+=1
            data_csv = {}
            for ch, ch_name in enumerate(channel_names):
                data = DATA[T][C][g][ch]
                if ch==0:
                    SOX2[T][C] = [*SOX2[T][C], *data]
                    data_csv["SOX2"] = data
                elif ch==1:
                    OCT4[T][C] = [*OCT4[T][C], *data]
                    data_csv["OCT4"] = data
                elif ch==2:
                    BRA[T][C] = [*BRA[T][C], *data]
                    data_csv["BRA"] = data
                elif ch==3:
                    DAPI[T][C] = [*DAPI[T][C], *data]
                    data_csv["DAPI"] = data
                
                ch_count+=1

            df = pd.DataFrame(data_csv)
            df.index.name = "cell_id"
            filename = path_results_c+embcode+".csv"
            df.to_csv(filename)
    
