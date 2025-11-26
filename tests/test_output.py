from ppath.pp import load_wrf_dataset

if __name__ == '__main__':
    source = r"D:\temp\AQMD\20251124_WRF_voxel_small.nc"
    wrf_dataset = load_wrf_dataset(source)