### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIME = "48hr"
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/stacks/{}/'.format(TIME)
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/ctobjects/{}/'.format(TIME)
path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/"
try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)
    
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
DATA = []
CONDS = []
for file in files:
    channel_names = ["SOX2", "OCT4", "BRA", "DAPI"]

    file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
    if "DMSO" in file:
        CONDS.append("DMSO")
    elif "CHIR" in file:
        CONDS.append("CHIRON")
    
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
        'blur': [5,1], 
    }

    concatenation3D_args = {
        'distance_th_z': 3.0, # microns
        'relative_overlap':False, 
        'use_full_matrix_to_compute_overlap':True, 
        'z_neighborhood':2, 
        'overlap_gradient_th':0.3, 
        'min_cell_planes': 3,
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
        'plot_centers':[True, True], # [Plot center as a dot, plot label on 3D center]
        # 'channels':[ch],
        'channels': chans_plot,
        'min_outline_length':75,
    }

    CT = CellTracking(
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
    # CT.plot_tracking(plot_args)

    # from embdevtools import remove_small_cells, plot_cell_sizes
    # plot_cell_sizes(CT, bw=30, bins=40, xlim=(0,400))
    # remove_small_cells(CT, 57, update_labels=True)

    from embdevtools import get_intenity_profile
    ch = channel_names.index("DAPI")
    correction_function, intensity_profile, z_positions = get_intenity_profile(CT, ch)

    # CT.plot_tracking(plot_args)
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
    DATA.append(data)
    

colors = np.asarray([[1.0, 0.0, 1.0, 0.7],
          [1.0, 0.0, 1.0, 1.0],
          [0.9, 0.9, 0.0, 0.7],
          [0.9, 0.9, 0.0, 1.0],
          [0.0, 0.8, 0.0, 0.7],
          [0.0, 0.8, 0.0, 1.0],
          [0.5, 0.5, 0.5, 0.7],
          [0.5, 0.5, 0.5, 1.0]])

DATA.append([[],[],[],[]])
channel_names_plot = []
data = []

for ch, ch_name in enumerate(channel_names):
    channel_names_plot.append("{}-D".format(ch_name))
    channel_names_plot.append("{}-C".format(ch_name))
    data.append(DATA[0][ch])
    data.append(DATA[1][ch])
data_means = [np.mean(d) for d in data]
data_stds  = [np.std(d) for d in data]


fig, ax = plt.subplots()
ax.bar(range(1,9), data_means, tick_label=channel_names_plot, yerr=data_stds, color=colors, capsize=6)
ax.set_title(TIME)
plt.tight_layout()
plt.savefig("{}{}".format(path_figures, TIME))
plt.show()

data = np.array(data)
for ch, ch_name in enumerate(channel_names):
    idx = ch*2.0
    print(idx)
    els = np.array([idx,idx+1], dtype="int32")
    print(els)
    bras = [val for d in data[els] for val in d]
    bra1 = data[els[0]]
    bra2 = data[els[1]]

    fig, ax =  plt.subplots()
    y1,x1, _ = ax.hist(bras, bins=50, density=False, alpha=0.5, label="mix")
    y2,x2, _ = ax.hist(bra1, bins=50, density=False, alpha=0.5,label="DMSO")
    y3,x3, _ = ax.hist(bra2, bins=50, density=False, alpha=0.5, label="CHIR")
    ax.legend()
    ax.set_title(ch_name)
    plt.savefig("{}{}_{}".format(path_figures, TIME, ch_name))
    plt.show()

