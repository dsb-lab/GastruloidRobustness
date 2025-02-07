from qlivecell import get_file_names, get_file_name
import czifile
import tifffile
import numpy as np

CONDITIONS = ["Wnt3KO_DMSO", "WT_CHIR", "WT_DMSO"]
CONDITIONS_48 = ["Wnt3KO", "WT"]

TIMES = ["48h", "60h", "72h", "84h", "96h"]
for TIME in TIMES:
    print()
    print(TIME)
    if TIME=="48h":
        CONDS = CONDITIONS_48
    else:
        CONDS = CONDITIONS

    for COND in CONDS:
        print(COND)
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/Sox2_Oct4_Bra_DAPI/{}/{}/'.format(TIME, COND)
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/stephen/Nanog_Cdx2_Otx2_DAPI/{}/{}/'.format(TIME, COND)

        _files = get_file_names(path_data_dir)
        files = [file for file in _files if ".czi" in file]
        for file in files:
            print(file)
            IMG = czifile.imread(path_data_dir+file)
            arr_squeezed = np.squeeze(IMG, axis=(1, 6)) 
            arr_final = arr_squeezed.swapaxes(1, 2)   
            print(IMG.shape)
            
            mdata = {"axes": "TZCYX", "spacing": 2, "unit": "um"}

            tifffile.imwrite(
                path_data_dir + file.split(".")[0]+".tif", arr_final, imagej=True, resolution=(0.3452677, 0.3452677), metadata=mdata
            )
