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
# TIMES = ["48h", "60h", "72h", "84h"]

CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]
channel_names = ["NANOG", "CDX2", "OTX2", "DAPI"]

files_exclude = [
    "G6-E14 48h SOX2 647 OCT4 546 BRA 488 DAPI_11.tif",
    # "G4-WNT3KO 60H DMSO SOX2 647 OCT4 555 BRA 488 DAPI_22.tif",
    "G2-E14 96h CHIR SOX2 647 OCT4 546 BRA 488 DAPI_17.tif",
    "G5-72h E14 CHIR NANOG647 CDX2_555 OTX2_488 DAPI_26.tif",
    "G4-E14 DMSO 96H NANOG_647 CDX2_555 OTX2_488 DAPI_20.tif",
    "G1-WNT3KO DMSO 96H NANOG_647 CDX2_555 OTX2_488 DAPI_31.tif",
    "G6-96h WNT3KO DMSO NANOG647 CDX2_555 OTX2_488 DAPI_26.tif",
    "G2-E14 60H DMSO SOX2 647 OCT4 555 BRA 488 DAPI_22.tif"
]
 
size_thresholds = [18.3, 14.4, 14.8, 12.5, 14.600000000000001]

n_cells = []
DATA = []

TIME = TIMES[0]
COND = CONDITIONS_48[1]

path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)
try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
file = files[4]
print(file)
if file in files_exclude:
    print("CHANGE FILE")
    

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
    # 'plot_stack_dims': (512, 512), 
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
# CT.plot(plot_args=plot_args)

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


fig ,ax = plt.subplots(, 4, sharex='row', sharey='row')

data = [[] for i in range(4)]
zs = []
for ch in range(CT.hyperstack.shape[2]):  
    for cell in CT.jitcells:
        z = int(cell.centers[0][0])
        img = CT.hyperstack[0, z, ch]
        zid = cell.zs[0].index(z)
        center = cell.centers[0][1:]
        mask = cell.masks[0][zid]  
        val = np.maximum(0, np.mean(img[mask[:, 1], mask[:, 0]]) - np.mean(img[0:50, 0:50]))
        data[ch].append(val)
        if ch==0:
            zs.append(z)

data1, data2 = remove_outliers_pairs(data[0], data[-1])
# ax[0].scatter(data1, data2, c=zs, s=3)
ax[1,0].scatter(data1, data2, s=3)
ax[0,0].hist(data1, bins=50, alpha=0.5)
ax[0,1].hist(data2, bins=50, alpha=0.5)
# correction_function, intensity_profile, z_positions = get_intenity_profile(CT, -1)
# plt.plot(correction_function)
# plt.show()

labs_to_rem = []
z_min = -1
data = [[] for i in range(4)]
zs = []
for ch in range(CT.hyperstack.shape[2]):
    correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
    if ch==3:
        z_min = np.argmax(intensity_profile)
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
        
for ch in range(CT.hyperstack.shape[2]):  
    for cell in CT.jitcells:
        z = int(cell.centers[0][0])
        img = CT.hyperstack[0, z, ch]
        zid = cell.zs[0].index(z)
        center = cell.centers[0][1:]
        mask = cell.masks[0][zid]  
        val = np.maximum(0, np.mean(img[mask[:, 1], mask[:, 0]]) - np.mean(img[0:50, 0:50]))
        data[ch].append(val)
        if ch==0:
            zs.append(z)
    

# save_4Dstack("/home/pablo/Desktop/", "masks", CT._masks_stack, 1/CT.CT_info.xyresolution, CT.CT_info.zresolution)

data1, data2 = remove_outliers_pairs(data[0], data[-1])
# ax[1].scatter(data1, data2, c=zs, s=3)
ax[1,1].scatter(data1, data2, s=3)

ax[0,0].hist(data1, bins=50, alpha=0.5)
ax[0,1].hist(data2, bins=50, alpha=0.5)
plt.show()

plt.plot(intensity_profile)
plt.show()