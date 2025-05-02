### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift, get_intenity_profile
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

exp=1
EXP = EXPERIMENTS[exp]
TIME = TIMES[3]

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/{}/".format(EXP)
try: 
    _ = get_file_names(path_figures)
except: 
    import os
    os.mkdir(path_figures)

if TIME=="48h":
    CONDS = CONDITIONS_48
else:
    CONDS = CONDITIONS

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

channel_names = CHANNEL_NAMES[exp]
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

DATA = []
for C, COND in enumerate(CONDS):
    print()
    print(COND)
    DATA.append([])
    path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/{}/{}/{}/'.format(EXP, TIME, COND)
    path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/{}/{}/'.format(EXP, TIME, COND)
    ### GET FULL FILE NAME AND FILE CODE ###
    files = get_file_names(path_data_dir)
    files = [file for file in files if ".tif" in file]
    print(files)
    try: 
        _ = get_file_names(path_save_dir)
    except: 
        import os
        os.mkdir(path_save_dir)
    
    for file in files:
        file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)

        path_data = path_data_dir+file
        path_save = path_save_dir+embcode
        
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


        ch = channel_names.index("DAPI")
        correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(z_positions, intensity_profile)
        # ax.plot(range(CT.slices), correction_function, ls="--")
        # plt.show()

        import numpy as np
        for ch in range(CT.hyperstack.shape[2]):
            _correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)
            stack = CT.hyperstack[0,:,ch].astype("float32")
            for z in range(stack.shape[0]):
                stack[z] = stack[z] / correction_function[z]
            stack *= np.mean(intensity_profile)
            CT.hyperstack[0,:,ch] = stack.astype("uint8")

        # CT.plot_tracking(plot_args)

        results = extract_fluoro(CT)

        import matplotlib.pyplot as plt
        data = []
        for ch_name in channel_names:
            ch = channel_names.index(ch_name)
            vals = results["channel_{}".format(ch)]
            data.append(vals)
        DATA[C].append(data)

import matplotlib as mpl
cmap = mpl.colormaps["tab20"]


data = []
channel_names_plot = []
    
for ch, ch_name in enumerate(channel_names):
    for C in range(len(CONDS)):
        data.append([])
        channel_names_plot.append("{}-{}".format(ch_name, CONDS[C]))
        for f, file in enumerate(files):
            data[-1] = [*data[-1], *DATA[C][f][ch]]

data_means = [np.mean(d) for d in data]
data_stds  = np.asarray([[0 for d in data], [np.std(d) for d in data]])

colors = []
for ch, ch_name in enumerate(channel_names):
    for C in range(len(CONDS)):
        colors.append(np.asarray(cmap(ch*2)))
        colors[-1][-1] = 1-C*0.4
        

fig, ax = plt.subplots(figsize=(15,5))
ax.bar(range(1,len(channel_names)*len(CONDS)-2), data_means[:-3], tick_label=channel_names_plot[:-3], yerr=data_stds[:, :-3], capsize=6, color=colors)
ax.set_title(TIME)
plt.tight_layout()
plt.savefig("{}{}".format(path_figures, TIME))
plt.show()

fig, ax = plt.subplots(figsize=(10,5))
violin_parts = ax.violinplot(data, showmeans=True, showmedians=True)
for pp, pc in enumerate(violin_parts['bodies']):
    
    pc.set_facecolor(colors[pp])
    pc.set_edgecolor('black')


# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
    vp = violin_parts[partname]
    vp.set_edgecolor("gray")
    vp.set_linewidth(1)

ax.set_title(TIME)
plt.tight_layout()
plt.savefig("{}{}".format(path_figures, TIME))
plt.show()

ch_g = 0
data = np.array(data)
for ch, ch_name in enumerate(channel_names):
    fig, ax =  plt.subplots()
    print(ch)
    for C, COND in enumerate(CONDS):
        if C!=0:continue
        id1 = ch*len(CONDS) + C
        ax.hist(data[id1], bins=200, density=False, alpha=0.5,label=channel_names_plot[ch_g])
        ch_g +=1
    ax.legend()
    # ax.set_yscale('log')
    ax.set_title(ch_name)
    plt.savefig("{}{}_{}".format(path_figures, TIME, ch_name))
    plt.show()
