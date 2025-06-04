### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, get_intenity_profile, get_file_names, construct_RGB, extract_fluoro, tif_reader_5D
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_save_figs = '/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/Nanog_Cdx2_Otx2_DAPI/'

EXP = "Nanog_Cdx2_Otx2_DAPI"

path_save_figs="/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/Sox2_Oct4_Bra_DAPI/"

EXP = "Sox2_Oct4_Bra_DAPI"

TIMES = ["48h", "60h", "72h"]
# TIMES = ["48h", "60h", "72h", "84h", "96h"]

CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]
channel_names = ["NANOG", "CDX2", "OTX2", "DAPI"]
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

T = 2
TIME = TIMES[T]
C = 1
if TIME=="48h":
    CONDS = CONDITIONS_48
else:
    CONDS = CONDITIONS
    
COND = CONDS[C]


path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)
try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
file = files[5]
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

nanog_pre = []
dapi_pre = []
for cell in CT.jitcells:
    z = int(cell.centers[0][0])
    zid = cell.zs[0].index(z)
    center = cell.centers[0][1:]
    mask = cell.masks[0][zid]  
    
    img = CT.hyperstack[0, z, 0]
    val = np.maximum(0, np.mean(img[mask[:, 1], mask[:, 0]]) - np.mean(img[0:50, 0:50]))
    nanog_pre.append(val)
    
    img = CT.hyperstack[0, z, -1]
    val = np.maximum(0, np.mean(img[mask[:, 1], mask[:, 0]]) - np.mean(img[0:50, 0:50]))
    dapi_pre.append(val)
    
fig, ax = plt.subplots(2,4,sharey='col', figsize=(15,8))

ch=2

ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[0, 2].axis('off')

ax[1, 0].axis('off')
ax[1, 1].axis('off')
ax[1, 2].axis('off')

ax[0, 0].imshow(CT.hyperstack[0,10,ch], vmin=0, vmax=50)
ax[0, 1].imshow(CT.hyperstack[0,45,ch], vmin=0, vmax=50)
ax[0, 2].imshow(CT.hyperstack[0,75,ch], vmin=0, vmax=50)

correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
z_min = z_positions[np.argmax(intensity_profile)]

stack = CT.hyperstack[0,:,ch].astype("float32")
for z in range(stack.shape[0]):
    stack[z] = stack[z] / correction_function[z]
stack *= np.mean(intensity_profile)
CT.hyperstack[0,:,ch] = stack.astype("uint8")

nanog_post = []
dapi_post = []
for cell in CT.jitcells:
    z = int(cell.centers[0][0])
    zid = cell.zs[0].index(z)
    center = cell.centers[0][1:]
    mask = cell.masks[0][zid]  
    
    img = CT.hyperstack[0, z, 0]
    val = np.maximum(0, np.mean(img[mask[:, 1], mask[:, 0]]) - np.mean(img[0:50, 0:50]))
    nanog_post.append(val)
    
    img = CT.hyperstack[0, z, -1]
    val = np.maximum(0, np.mean(img[mask[:, 1], mask[:, 0]]) - np.mean(img[0:50, 0:50]))
    dapi_post.append(val)
    
ax[0, 3].plot(z_positions, intensity_profile, label="intensity profile")
ax[0, 3].plot(correction_function, label="correction function")
ax[0, 3].legend()

correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
ax[1, 3].plot(z_positions, intensity_profile, label="corrected intensity profile")
ax[1, 3].legend()
ax[1, 0].imshow(CT.hyperstack[0,10,ch], vmin=0, vmax=50)
ax[1, 1].imshow(CT.hyperstack[0,45,ch], vmin=0, vmax=50)
ax[1, 2].imshow(CT.hyperstack[0,75,ch], vmin=0, vmax=50)

ax[0,3].set_ylabel("mean nuclear {}".format(channel_names[ch]))
ax[1,3].set_ylabel("mean nuclear {}".format(channel_names[ch]))

ax[0,3].set_xlabel("Z-position")
ax[1,3].set_xlabel("Z-position")

ax[0, 3].set_box_aspect(1)
ax[1, 3].set_box_aspect(1)

plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/z_correction/stacks_{}.svg".format(channel_names[ch]))
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/z_correction/stacks_{}.pdf".format(channel_names[ch]))

plt.show()


# Function to remove outliers using IQR method
def remove_outliers_pairs(data1, data2):
    # Calculate the first and third quartiles (Q1 and Q3)
    Q11 = np.percentile(data1, 25)
    Q31 = np.percentile(data1, 75)
    IQR1 = Q31 - Q11

    # Define the lower and upper bounds for outliers
    lower_bound1 = Q11 - 1.5 * IQR1
    upper_bound1 = Q31 + 1.5 * IQR1

    # Calculate the first and third quartiles (Q1 and Q3)
    Q12 = np.percentile(data2, 25)
    Q32 = np.percentile(data2, 75)
    IQR2 = Q32 - Q12

    # Define the lower and upper bounds for outliers
    lower_bound2 = Q12 - 1.5 * IQR2
    upper_bound2 = Q32 + 1.5 * IQR2

    final_data1 = []
    final_data2 = []
    # Filter data to remove outliers
    for i in range(len(data1)):
        if lower_bound1 <= data1[i] <= upper_bound1:
            if lower_bound2 <= data2[i] <= upper_bound2:
                final_data1.append(data1[i])
                final_data2.append(data2[i])
    return final_data1, final_data2

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].set_title("uncorrected")
ax[1].set_title("corrected")

# Pre data
data1 = dapi_pre
data2 = nanog_pre
data1, data2 = remove_outliers_pairs(data1, data2)
data12 = np.vstack([data1, data2])
cols1 = gaussian_kde(data12)(data12)
ax[0].scatter(data1, data2, s=1, c=cols1)
ax[0].set_xlabel("DAPI")
ax[0].set_ylabel("NANOG")

# Post data
data1 = dapi_post
data2 = nanog_post
data1, data2 = remove_outliers_pairs(data1, data2)
data12 = np.vstack([data1, data2])
cols2 = gaussian_kde(data12)(data12)
ax[1].scatter(data1, data2, s=1, c=cols2)
ax[1].set_xlabel("DAPI")

# Combine colors to normalize
all_cols = np.hstack([cols1, cols2])
norm = Normalize(vmin=np.min(all_cols), vmax=np.max(all_cols))
sm = ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Add colorbar manually to the right
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([norm.vmin, norm.vmax])
cbar.set_ticklabels(['low', 'high'])

# Move the label closer to the colorbar
cbar.set_label("Point Density", labelpad=-10)  # Reduce labelpad for closer positioning

plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space on right for colorbar
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/z_correction/scatter_example.svg")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/z_correction/scatter_example.pdf")

plt.show()
