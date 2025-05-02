### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, get_intenity_profile, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')
### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
EXPERIMENTS = ["Sox2_Oct4_Bra_DAPI", "Nanog_Cdx2_Otx2_DAPI"]
CH_NAMES = [["SOX2", "OCT4", "BRA", "DAPI"], ["NANOG", "CDX2", "OTX2", "DAPI"]]
TIMES = ["48h", "60h", "72h", "84h", "96h"]
CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]

areas = []

for E, EXP in enumerate(EXPERIMENTS):
    channel_names = CH_NAMES[E]
    for TIME in TIMES:
        if E==0:
            areas.append([])

        if TIME=="48h":
            CONDS = CONDITIONS_48
        else:
            CONDS = CONDITIONS

        for COND in CONDS:
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
                
                for cell in CT.jitcells:
                    zc = int(cell.centers[0][0])
                    zcid = cell.zs[0].index(zc)

                    msk = cell.masks[0][zcid]
                    area = len(msk)
                    areas[-1].append(area)
                
  
import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=14) 
mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 
mpl.rc('legend', fontsize=14) 

from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity


fig, ax = plt.subplots(2,len(TIMES), figsize=(14,4))
thresholds = []
for T in range(len(TIMES)):
    data = np.array(areas[T])/CT.metadata["XYresolution"]**2
    ax[0, T].hist(data, bins=200, color=[0.0, 0.8, 0.0], density=True, alpha=0.6, label=TIMES[T])

    x = np.arange(0, step=0.1, stop=np.max(data))
    bw = 5
    modelo_kde = KernelDensity(kernel="linear", bandwidth=bw)
    modelo_kde.fit(X=data.reshape(-1, 1))
    densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1, 1))))
    ax[0, T].plot(x, densidad_pred, color="magenta")

    local_minima = argrelextrema(densidad_pred, np.less)[0]
    thresholds.append(x[local_minima[0]])
    x_th = np.ones(len(x)) * x[local_minima[0]]
    y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
    ax[0, T].plot(x_th, y_th, c="k", ls="--",lw=2, label="debris th.")

    ax[0, T].set_ylabel("count")
    ax[1, T].set_ylabel("count")

    ax[0, T].set_title(TIMES[T])
    ax[1, T].hist(data, bins=200, color=[0.0, 0.8, 0.0], density=True, alpha=0.6, label=TIMES[T])
    ax[1, T].set_xlabel(r"area ($\mu$m$^2$)")
    ax[1, T].plot(x, densidad_pred, color="magenta")
    ax[1, T].plot(x_th, y_th, c="k", ls="--",lw=2, label="debris th.")

    ax[1, T].set_xlim(-1, 75)
plt.tight_layout()

plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/debris/debris_thresholds.svg")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidRobustness/figures/debris/debris_thresholds.pdf")
plt.show()

