import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
import re
import pyproj


def calculate_hours_since_base(dt_obj, base_dt=datetime(1900, 1, 1)):
    """Helper to convert datetime to float hours since base date."""
    delta = dt_obj - base_dt
    return delta.total_seconds() / 3600.0


def load_wrf_dataset(file_paths):
    """
    Reads multiple WRF NetCDF files and concatenates them along the 'Time' dimension.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    try:
        ds = xr.open_mfdataset(
            sorted(file_paths),
            concat_dim='Time',
            combine='nested',
            parallel=False,
            chunks={'Time': 1}
        )

        ds = ds.sortby('Times', ascending=True)

        print(f"Successfully loaded {len(file_paths)} WRF files.")
        return ds
    except Exception as e:
        print(f"Error loading multiple WRF files: {e}")
        raise


def load_cmaq_dataset(file_paths):
    """
    Reads multiple CMAQ (ACONC/APMDIAG) NetCDF files, concatenates them along
    the 'TSTEP' dimension, and sorts the result by the TFLAG variable.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    try:
        ds = xr.open_mfdataset(
            sorted(file_paths),
            concat_dim='TSTEP',
            combine='nested',
            parallel=False,
            chunks={'TSTEP': 1}
        )

        if 'TFLAG' in ds:
            tflag_ref = ds['TFLAG'].isel(VAR=0)
            dates = tflag_ref.isel(**{'DATE-TIME': 0})
            times = tflag_ref.isel(**{'DATE-TIME': 1})
            sort_key = (dates.astype('int64') * 1000000) + times.astype('int64')
            ds = ds.sortby(sort_key)
            print("Dataset sorted by TFLAG.")

        print(f"Successfully loaded {len(file_paths)} CMAQ files.")
        return ds

    except Exception as e:
        print(f"Error loading multiple CMAQ files: {e}")
        raise


def calculate_new_variables(calc_config, loaded_datasets):
    """
    Calculates new variables based on a list of formulas and loaded datasets.
    """
    calculated_vars = {}
    print("\n--- Calculating Variables ---")

    for calc_item in calc_config:
        var_name = calc_item.get("var")
        formula = calc_item.get("formula")
        unit = calc_item.get("unit")

        if not var_name or not formula:
            continue

        print(f"Processing: {var_name}")

        def replace_ref(match):
            lbl = match.group(1)
            var = match.group(2)

            if lbl in loaded_datasets:
                if var in loaded_datasets[lbl]:
                    return f"loaded_datasets['{lbl}']['{var}']"
                else:
                    raise ValueError(f"Variable '{var}' not found in dataset '{lbl}'")
            else:
                return match.group(0)

        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b"

        try:
            executable_formula = re.sub(pattern, replace_ref, formula)
            result_da = eval(executable_formula, {"loaded_datasets": loaded_datasets, "np": np})

            result_da.name = var_name
            if unit:
                result_da.attrs['units'] = unit

            calculated_vars[var_name] = result_da

        except Exception as e:
            print(f"Error calculating '{var_name}': {e}")

    return calculated_vars


def process_wrf_netcdf(input_source, output_path, variables_to_keep=None, layers_to_keep=None, epsg=102100):
    """
    Converts WRF NetCDF file(s) into an ArcGIS Voxel Layer compliant NetCDF.
    Handles coordinate transformation based on EPSG.
    """

    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_wrf_dataset(input_source)

    # --- 1. Determine Target CRS ---
    target_crs = pyproj.CRS.from_epsg(epsg)
    is_geographic = target_crs.is_geographic

    # --- 2. Generate Spatial Coordinates ---
    nx = ds.sizes['west_east']
    ny = ds.sizes['south_north']
    nz = ds.sizes['bottom_top']

    # Extract 2D Lat/Lon from WRF (WGS84)
    try:
        lat_2d = ds['XLAT'].isel(Time=0).values
        lon_2d = ds['XLONG'].isel(Time=0).values
    except KeyError:
        print("Error: XLAT or XLONG not found. Cannot calculate coordinates.")
        return

    # Create Transformer: WRF Lat/Lon (EPSG:4326) -> Target EPSG
    transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    # Transform grid points to target CRS
    xx_target, yy_target = transformer.transform(lon_2d, lat_2d)

    # Calculate average spacing and origin in target CRS
    # Axis 0 is south_north (y), Axis 1 is west_east (x)
    dy_target = np.mean(np.diff(yy_target, axis=0))
    dx_target = np.mean(np.diff(xx_target, axis=1))

    y0 = yy_target[0, 0]
    x0 = xx_target[0, 0]

    # Create 1D arrays
    x_coords = (x0 + np.arange(nx) * dx_target).astype('float32')
    y_coords = (y0 + np.arange(ny) * dy_target).astype('float32')
    z_coords = np.arange(nz, dtype='float32')

    # --- 3. Generate Time Coordinate ---
    base_date = datetime(1900, 1, 1)

    if 'Times' in ds.variables:
        raw_times = ds['Times'].values
        time_values = []
        for t in raw_times:
            try:
                t_str = t.decode('utf-8')
                dt = datetime.strptime(t_str, '%Y-%m-%d_%H:%M:%S')
                time_values.append(calculate_hours_since_base(dt, base_date))
            except Exception:
                time_values.append(0.0)
        time_coords = np.array(time_values, dtype='float32')
    else:
        time_coords = np.arange(ds.sizes['Time'], dtype='float32')

    # --- 4. Create New Compliant Dataset ---
    new_ds = xr.Dataset()

    new_ds = new_ds.assign_coords({
        'time': time_coords,
        'level': z_coords,
        'y': y_coords,
        'x': x_coords
    })

    # Set CF/ArcGIS Attributes for Coordinates
    if is_geographic:
        new_ds['x'].attrs = {
            'standard_name': 'longitude',
            'units': 'degrees_east',
            'axis': 'X',
            '_FillValue': -999.0
        }
        new_ds['y'].attrs = {
            'standard_name': 'latitude',
            'units': 'degrees_north',
            'axis': 'Y',
            '_FillValue': -999.0
        }
    else:
        new_ds['x'].attrs = {
            'standard_name': 'projection_x_coordinate',
            'units': 'm',
            'axis': 'X',
            '_FillValue': -999.0
        }
        new_ds['y'].attrs = {
            'standard_name': 'projection_y_coordinate',
            'units': 'm',
            'axis': 'Y',
            '_FillValue': -999.0
        }

    new_ds['level'].attrs = {
        'standard_name': 'model_level_number',
        'units': 'level',
        'positive': 'up',
        'axis': 'Z',
        '_FillValue': -999.0
    }
    new_ds['time'].attrs = {
        'standard_name': 'time',
        'units': 'hours since 1900-01-01 00:00:00',
        'calendar': 'gregorian',
        'axis': 'T',
        '_FillValue': -999.0
    }

    # Add ESRI PE String for ArcGIS Pro recognition
    try:
        esri_pe = target_crs.to_wkt()
        new_ds.attrs['esri_pe_string'] = esri_pe
    except Exception:
        pass

    # --- 5. Transfer and Destagger Data Variables ---
    dim_map = {
        'Time': 'time',
        'bottom_top': 'level',
        'south_north': 'y',
        'west_east': 'x'
    }

    if variables_to_keep:
        vars_list = [v for v in variables_to_keep if v in ds.variables]
    else:
        exclude = ['Times', 'XLAT', 'XLONG', 'XTIME', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V']
        vars_list = [v for v in ds.data_vars if v not in exclude]

    for var_name in vars_list:
        src_var = ds[var_name]
        values = src_var.values
        dims = list(src_var.sizes)

        # Destaggering Logic
        if 'west_east_stag' in dims:
            axis_idx = dims.index('west_east_stag')
            sl_left = [slice(None)] * values.ndim
            sl_right = [slice(None)] * values.ndim
            sl_left[axis_idx] = slice(None, -1)
            sl_right[axis_idx] = slice(1, None)
            values = 0.5 * (values[tuple(sl_left)] + values[tuple(sl_right)])
            dims[axis_idx] = 'west_east'

        if 'south_north_stag' in dims:
            axis_idx = dims.index('south_north_stag')
            sl_left = [slice(None)] * values.ndim
            sl_right = [slice(None)] * values.ndim
            sl_left[axis_idx] = slice(None, -1)
            sl_right[axis_idx] = slice(1, None)
            values = 0.5 * (values[tuple(sl_left)] + values[tuple(sl_right)])
            dims[axis_idx] = 'south_north'

        if 'bottom_top_stag' in dims:
            axis_idx = dims.index('bottom_top_stag')
            sl_left = [slice(None)] * values.ndim
            sl_right = [slice(None)] * values.ndim
            sl_left[axis_idx] = slice(None, -1)
            sl_right[axis_idx] = slice(1, None)
            values = 0.5 * (values[tuple(sl_left)] + values[tuple(sl_right)])
            dims[axis_idx] = 'bottom_top'

        new_dims = [dim_map.get(d, d) for d in dims]

        if all(d in new_ds.coords for d in new_dims):
            print(f"  Converting variable: {var_name}")
            new_ds[var_name] = (new_dims, values)
            attrs = src_var.attrs.copy()
            if 'coordinates' in attrs: del attrs['coordinates']
            if 'stagger' in attrs: del attrs['stagger']
            new_ds[var_name].attrs = attrs
            new_ds[var_name].attrs['_FillValue'] = -999.0

    # --- 6. Filter Vertical Layers ---
    if layers_to_keep is not None:
        print(f"Filtering output to {len(layers_to_keep)} selected layers...")
        try:
            new_ds = new_ds.sel(level=layers_to_keep * 2 if len(layers_to_keep) == 1 else layers_to_keep)
        except KeyError as e:
            print(f"Error: Selected layers {layers_to_keep} are out of bounds. {e}")
            raise

    # --- 7. Write Output ---
    print(f"Writing to {output_path}...")
    new_ds.to_netcdf(output_path, format='NETCDF4')
    print("Done.")


def process_cmaq_netcdf(input_source, output_path, variables_to_keep=None, layers_to_keep=None, epsg=102100):
    """
    Converts CMAQ (ACONC/APMDIAG) NetCDF file(s) into an ArcGIS Voxel Layer compliant NetCDF.
    Handles coordinate transformation based on EPSG.
    """

    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_cmaq_dataset(input_source)

    # --- 1. Determine Target CRS ---
    target_crs = pyproj.CRS.from_epsg(epsg)
    is_geographic = target_crs.is_geographic

    # --- 2. Generate Spatial Coordinates ---
    try:
        # Define Source Projection (CMAQ LCC)
        p_alp = ds.attrs.get('P_ALP')
        p_bet = ds.attrs.get('P_BET')
        p_gam = ds.attrs.get('P_GAM')
        y_cent = ds.attrs.get('YCENT')
        x_orig = ds.attrs.get('XORIG', 0.0)
        y_orig = ds.attrs.get('YORIG', 0.0)
        x_cell = ds.attrs.get('XCELL', 1000.0)
        y_cell = ds.attrs.get('YCELL', 1000.0)

        ncol = ds.sizes['COL']
        nrow = ds.sizes['ROW']
        nlay = ds.sizes['LAY']

        # CMAQ Proj String
        source_proj_str = (f"+proj=lcc +lat_1={p_alp} +lat_2={p_bet} +lat_0={y_cent} "
                           f"+lon_0={p_gam} +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs")

        source_crs = pyproj.CRS.from_proj4(source_proj_str)

        # Create 1D arrays in Source Meters
        x_source_m = x_orig + (np.arange(ncol) * x_cell)
        y_source_m = y_orig + (np.arange(nrow) * y_cell)
        xx_source, yy_source = np.meshgrid(x_source_m, y_source_m)

        # Transformer: Source -> Target EPSG
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        xx_target, yy_target = transformer.transform(xx_source, yy_source)

        # Calculate average spacing and origin in target CRS
        dy_target = np.mean(np.diff(yy_target, axis=0))
        dx_target = np.mean(np.diff(xx_target, axis=1))
        y0 = yy_target[0, 0]
        x0 = xx_target[0, 0]

        # Generate 1D Coordinates in Target Units
        x_coords = (x0 + np.arange(ncol) * dx_target).astype('float32')
        y_coords = (y0 + np.arange(nrow) * dy_target).astype('float32')
        z_coords = np.arange(nlay, dtype='float32')

    except Exception as e:
        print(f"Error calculating coordinates from projection: {e}")
        print("Falling back to generic index coordinates.")
        x_coords = np.arange(ds.sizes['COL'], dtype='float32')
        y_coords = np.arange(ds.sizes['ROW'], dtype='float32')
        z_coords = np.arange(ds.sizes['LAY'], dtype='float32')

    # --- 3. Generate Time Coordinate ---
    base_date = datetime(1900, 1, 1)

    if 'TFLAG' in ds.variables:
        tflag = ds['TFLAG'].values
        dates = tflag[:, 0, 0]
        times = tflag[:, 0, 1]

        time_values = []
        for d, t in zip(dates, times):
            d_str = str(d)
            t_str = str(t).zfill(6)
            try:
                dt_obj = datetime.strptime(f"{d_str}{t_str}", "%Y%j%H%M%S")
                time_values.append(calculate_hours_since_base(dt_obj, base_date))
            except ValueError:
                time_values.append(0.0)
        time_coords = np.array(time_values, dtype='float32')
    else:
        time_coords = np.arange(ds.sizes['TSTEP'], dtype='float32')

    # --- 4. Create New Compliant Dataset ---
    new_ds = xr.Dataset()

    new_ds = new_ds.assign_coords({
        'time': time_coords,
        'level': z_coords,
        'y': y_coords,
        'x': x_coords
    })

    # Set CF/ArcGIS Attributes
    if is_geographic:
        new_ds['x'].attrs = {
            'standard_name': 'longitude',
            'units': 'degrees_east',
            'axis': 'X',
            '_FillValue': -999.0
        }
        new_ds['y'].attrs = {
            'standard_name': 'latitude',
            'units': 'degrees_north',
            'axis': 'Y',
            '_FillValue': -999.0
        }
    else:
        new_ds['x'].attrs = {
            'standard_name': 'projection_x_coordinate',
            'units': 'm',
            'axis': 'X',
            '_FillValue': -999.0
        }
        new_ds['y'].attrs = {
            'standard_name': 'projection_y_coordinate',
            'units': 'm',
            'axis': 'Y',
            '_FillValue': -999.0
        }

    new_ds['level'].attrs = {
        'standard_name': 'model_level_number',
        'units': 'level',
        'positive': 'up',
        'axis': 'Z',
        '_FillValue': -999.0
    }
    new_ds['time'].attrs = {
        'standard_name': 'time',
        'units': 'hours since 1900-01-01 00:00:00',
        'calendar': 'gregorian',
        'axis': 'T',
        '_FillValue': -999.0
    }

    # Add ESRI PE String for ArcGIS Pro recognition
    try:
        esri_pe = target_crs.to_wkt()
        new_ds.attrs['esri_pe_string'] = esri_pe
    except Exception:
        pass

    # --- 5. Transfer Data Variables ---
    dim_map = {'TSTEP': 'time', 'LAY': 'level', 'ROW': 'y', 'COL': 'x'}

    if variables_to_keep:
        vars_list = [v for v in variables_to_keep if v in ds.variables]
    else:
        vars_list = [v for v in ds.data_vars if v != 'TFLAG']

    for var_name in vars_list:
        src_var = ds[var_name]
        new_dims = [dim_map.get(d, d) for d in src_var.sizes]

        if all(d in new_ds.coords for d in new_dims):
            print(f"  Converting variable: {var_name}")
            new_ds[var_name] = (new_dims, src_var.values)
            new_ds[var_name].attrs = src_var.attrs
            new_ds[var_name].attrs['_FillValue'] = -999.0

    # --- 6. Filter Vertical Layers ---
    if layers_to_keep is not None:
        print(f"Filtering output to {len(layers_to_keep)} selected layers...")
        try:
            new_ds = new_ds.sel(level=layers_to_keep * 2 if len(layers_to_keep) == 1 else layers_to_keep)
        except KeyError as e:
            print(f"Error: Selected layers {layers_to_keep} are out of bounds. {e}")
            raise

    # --- 7. Write Output ---
    print(f"Writing to {output_path}...")
    new_ds.to_netcdf(output_path, format='NETCDF4')
    print("Done.")


def extract_wrf_timeseries(input_source, output_path, station_cfg, variables_to_keep=None):
    """
    Extracts time-series data from WRF NetCDF files (or Dataset) for specific grid cells.
    """
    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_wrf_dataset(input_source)

    print(f"\nProcessing Time Series Output: {output_path}")

    if variables_to_keep:
        valid_vars = [v for v in variables_to_keep if v in ds.variables]
    else:
        exclude = ['Times', 'XLAT', 'XLONG', 'XTIME', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V']
        valid_vars = [v for v in ds.data_vars if v not in exclude]

    dt_index = None
    if 'Times' in ds:
        try:
            raw_times = ds['Times'].values
            dt_list = []
            for t in raw_times:
                if isinstance(t, bytes):
                    t_str = t.decode('utf-8')
                else:
                    t_str = str(t)
                dt_list.append(datetime.strptime(t_str, '%Y-%m-%d_%H:%M:%S'))
            dt_index = np.array(dt_list)
        except Exception as e:
            print(f"Warning: Could not parse 'Times' variable. Error: {e}")

    if dt_index is None and 'XTIME' in ds:
        dt_index = ds['XTIME'].values

    processed_vars = {}
    stag_map = {
        'west_east_stag': 'west_east',
        'south_north_stag': 'south_north',
        'bottom_top_stag': 'bottom_top'
    }

    print("Preparing variables (Lazy Destaggering)...")

    for var_name in valid_vars:
        da = ds[var_name]
        dims = da.dims

        for stag_dim, mass_dim in stag_map.items():
            if stag_dim in dims:
                da = 0.5 * (da.isel({stag_dim: slice(0, -1)}) +
                            da.isel({stag_dim: slice(1, None)}))
                da = da.rename({stag_dim: mass_dim})

        processed_vars[var_name] = da

    working_ds = xr.Dataset(processed_vars)

    valid_stations = []
    we_idxs = []
    sn_idxs = []
    bt_idxs = []
    labels = []

    for i, stn in enumerate(station_cfg):
        lbl = stn.get("label", f"Station_{i}")
        we = int(stn.get("west_east", 0))
        sn = int(stn.get("south_north", 0))
        bt = int(stn.get("bottom_top", 0))

        if (we < ds.sizes['west_east'] and
                sn < ds.sizes['south_north'] and
                bt < ds.sizes['bottom_top']):

            we_idxs.append(we)
            sn_idxs.append(sn)
            bt_idxs.append(bt)
            labels.append(lbl)
            valid_stations.append(stn)
        else:
            print(f"Skipping {lbl}: Indices out of bounds.")

    if not valid_stations:
        print(f"No valid stations found for {output_path}")
        return

    idx_station = xr.DataArray(np.arange(len(valid_stations)), dims="station")
    idx_we = xr.DataArray(we_idxs, dims="station")
    idx_sn = xr.DataArray(sn_idxs, dims="station")
    idx_bt = xr.DataArray(bt_idxs, dims="station")

    try:
        print(f"Extracting data for {len(valid_stations)} stations...")
        subset = working_ds.isel(
            west_east=idx_we,
            south_north=idx_sn,
            bottom_top=idx_bt
        )
        subset_loaded = subset.compute()
        df = subset_loaded.to_dataframe()
        df = df.reset_index()

        meta_df = pd.DataFrame({
            'station': np.arange(len(valid_stations)),
            'Station': labels,
            'west_east': we_idxs,
            'south_north': sn_idxs,
            'bottom_top': bt_idxs
        })

        final_df = pd.merge(df, meta_df, on='station', how='left')

        if dt_index is not None:
            if 'Time' in final_df.columns:
                max_t = final_df['Time'].max()
                if max_t < len(dt_index):
                    final_df['DateTime'] = dt_index[final_df['Time'].values]
                else:
                    final_df['DateTime'] = final_df['Time']
            else:
                final_df['DateTime'] = final_df.index

        meta_cols = ['Station', 'DateTime', 'west_east', 'south_north', 'bottom_top']
        data_cols = [c for c in valid_vars if c in final_df.columns]
        avail_meta = [c for c in meta_cols if c in final_df.columns]
        final_cols = avail_meta + data_cols

        final_df = final_df[final_cols]
        final_df = final_df.sort_values(by=avail_meta)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Saved time series to {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()


def extract_cmaq_timeseries(input_source, output_path, station_cfg, variables_to_keep=None):
    """
    Extracts time-series data from CMAQ NetCDF files (or Dataset) for specific grid cells.
    """
    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_cmaq_dataset(input_source)

    print(f"\nProcessing Time Series Output: {output_path}")

    if variables_to_keep:
        valid_vars = [v for v in variables_to_keep if v in ds.variables]
    else:
        valid_vars = [v for v in ds.data_vars if v != 'TFLAG']

    dt_index = None
    if 'TFLAG' in ds:
        try:
            tflag = ds['TFLAG'].isel(VAR=0).values
            dates = tflag[:, 0]
            times = tflag[:, 1]
            dt_list = []
            for d, t in zip(dates, times):
                d_str = str(d)
                t_str = str(t).zfill(6)
                try:
                    dt_list.append(datetime.strptime(f"{d_str}{t_str}", "%Y%j%H%M%S"))
                except ValueError:
                    dt_list.append(None)
            dt_index = np.array(dt_list)
        except Exception as e:
            print(f"Warning: Could not parse 'TFLAG'. Error: {e}")

    if dt_index is None and 'time' in ds.coords:
        dt_index = ds['time'].values

    valid_stations = []
    rows = []
    cols = []
    lays = []
    labels = []

    for i, stn in enumerate(station_cfg):
        r = int(stn.get("row", 0))
        c = int(stn.get("col", 0))
        l = int(stn.get("lay", 0))
        lbl = stn.get("label", f"Station_{i}")

        if (r < ds.sizes['ROW'] and c < ds.sizes['COL'] and l < ds.sizes['LAY']):
            rows.append(r)
            cols.append(c)
            lays.append(l)
            labels.append(lbl)
            valid_stations.append(stn)
        else:
            print(f"Skipping {lbl}: Indices out of bounds.")

    if not valid_stations:
        print(f"No valid stations found for {output_path}")
        return

    idx_station = xr.DataArray(np.arange(len(valid_stations)), dims="station")
    idx_row = xr.DataArray(rows, dims="station")
    idx_col = xr.DataArray(cols, dims="station")
    idx_lay = xr.DataArray(lays, dims="station")

    try:
        print(f"Extracting data for {len(valid_stations)} stations...")
        subset = ds[valid_vars].isel(ROW=idx_row, COL=idx_col, LAY=idx_lay)
        subset_loaded = subset.compute()
        df = subset_loaded.to_dataframe()
        df = df.reset_index()

        stn_meta = pd.DataFrame({
            'station': np.arange(len(valid_stations)),
            'Station': labels,
            'Row': rows,
            'Col': cols,
            'Layer': lays
        })

        final_df = pd.merge(df, stn_meta, on='station', how='left')

        if dt_index is not None:
            if len(dt_index) == ds.sizes['TSTEP']:
                n_timesteps = len(dt_index)
                n_stations = len(valid_stations)
                if 'TSTEP' in final_df.columns:
                    final_df = final_df.sort_values(by=['station', 'TSTEP'])
                    if len(final_df) == n_timesteps * n_stations:
                        final_df['DateTime'] = np.tile(dt_index, n_stations)
                    else:
                        final_df['DateTime'] = final_df['TSTEP']
                elif 'time' in final_df.columns:
                    final_df['DateTime'] = final_df['time']
            else:
                final_df['DateTime'] = final_df.index

        meta_cols = ['Station', 'DateTime', 'Layer', 'Row', 'Col']
        data_cols = [c for c in valid_vars if c in final_df.columns]
        avail_meta = [c for c in meta_cols if c in final_df.columns]
        final_cols = avail_meta + data_cols

        final_df = final_df[final_cols]
        final_df = final_df.sort_values(by=avail_meta)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Saved time series to {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()


def execute_cmaq_pipeline(json_input):
    """
    Orchestrates the CMAQ processing pipeline.
    """
    if isinstance(json_input, dict):
        config = json_input
    elif isinstance(json_input, str):
        if os.path.isfile(json_input):
            with open(json_input, 'r') as f:
                config = json.load(f)
        else:
            config = json.loads(json_input)
    else:
        raise ValueError("Input must be a dictionary, a JSON string, or a file path.")

    loaded_datasets = {}
    reference_tflag = None
    reference_attrs = None

    print("\n=== Starting CMAQ Pipeline ===")

    for source in config.get("source", []):
        label = source["label"]
        files = source["input_nc"]
        print(f"Loading source: {label}")

        ds = load_cmaq_dataset(files)

        if reference_tflag is None and 'TFLAG' in ds:
            reference_tflag = ds['TFLAG'].isel(VAR=slice(0, 1))
            reference_attrs = ds.attrs

        if 'TFLAG' in ds:
            ds = ds.drop_vars('TFLAG')
        if 'VAR' in ds.dims:
            ds = ds.drop_dims('VAR')

        loaded_datasets[label] = ds

    existing_vars = set()
    for ds in loaded_datasets.values():
        existing_vars.update(ds.data_vars)

    for calc in config.get("calc_var", []):
        if calc["var"] in existing_vars:
            raise ValueError(
                f"Calculated variable name '{calc['var']}' conflicts with an existing variable.")

    calculated_vars_dict = calculate_new_variables(config.get("calc_var", []), loaded_datasets)
    calc_ds = xr.Dataset(calculated_vars_dict)

    datasets_to_merge = list(loaded_datasets.values()) + [calc_ds]

    print("\nMerging datasets...")
    dataset_a = xr.merge(datasets_to_merge, compat='override')

    if reference_attrs:
        dataset_a.attrs = reference_attrs

    for v_out in config.get("voxel_ouput", []):
        out_path = v_out["output_nc"]
        sel_vars = v_out["selected_var"]
        sel_lay = v_out.get("selected_lay", None)
        # Get EPSG, default to 102100 if missing
        epsg_code = v_out.get("epsg", 102100)

        print(f"\nProcessing Voxel Output: {out_path} (EPSG: {epsg_code})")

        if reference_tflag is not None:
            dataset_a['TFLAG'] = reference_tflag

        process_cmaq_netcdf(dataset_a, out_path, variables_to_keep=sel_vars, layers_to_keep=sel_lay, epsg=epsg_code)

    for ts_out in config.get("ts_ouput", []):
        out_csv = ts_out["output_csv"]
        sel_vars = ts_out["selected_var"]
        stations = ts_out["station"]

        if reference_tflag is not None:
            dataset_a['TFLAG'] = reference_tflag

        extract_cmaq_timeseries(
            input_source=dataset_a,
            output_path=out_csv,
            station_cfg=stations,
            variables_to_keep=sel_vars
        )

    print("\n=== Pipeline Completed ===")


def execute_wrf_pipeline(json_input):
    """
    Orchestrates the WRF processing pipeline.
    """
    if isinstance(json_input, dict):
        config = json_input
    elif isinstance(json_input, str):
        if os.path.isfile(json_input):
            with open(json_input, 'r') as f:
                config = json.load(f)
        else:
            config = json.loads(json_input)
    else:
        raise ValueError("Input must be a dictionary, a JSON string, or a file path.")

    loaded_datasets = {}
    print("\n=== Starting WRF Pipeline ===")

    for source in config.get("source", []):
        label = source["label"]
        files = source["input_nc"]
        print(f"Loading source: {label}")

        ds = load_wrf_dataset(files)
        loaded_datasets[label] = ds

    existing_vars = set()
    for ds in loaded_datasets.values():
        existing_vars.update(ds.data_vars)

    for calc in config.get("calc_var", []):
        if calc["var"] in existing_vars:
            raise ValueError(
                f"Calculated variable name '{calc['var']}' conflicts with an existing variable.")

    calculated_vars_dict = calculate_new_variables(config.get("calc_var", []), loaded_datasets)
    calc_ds = xr.Dataset(calculated_vars_dict)

    print("\nMerging datasets...")
    datasets_to_merge = list(loaded_datasets.values()) + [calc_ds]
    dataset_a = xr.merge(datasets_to_merge, compat='override')

    for v_out in config.get("voxel_ouput", []):
        out_path = v_out["output_nc"]
        sel_vars = v_out["selected_var"]
        sel_lay = v_out.get("selected_lay", None)
        # Get EPSG, default to 102100 if missing
        epsg_code = v_out.get("epsg", 102100)

        print(f"\nProcessing Voxel Output: {out_path} (EPSG: {epsg_code})")

        valid_vars = [v for v in sel_vars if v in dataset_a]
        missing_vars = [v for v in sel_vars if v not in dataset_a]

        if missing_vars:
            print(f"Warning: The following variables were not found and skipped: {missing_vars}")

        process_wrf_netcdf(dataset_a, out_path, variables_to_keep=valid_vars, layers_to_keep=sel_lay, epsg=epsg_code)

    for ts_out in config.get("ts_ouput", []):
        out_csv = ts_out["output_csv"]
        sel_vars = ts_out["selected_var"]
        stations = ts_out["station"]

        extract_wrf_timeseries(
            input_source=dataset_a,
            output_path=out_csv,
            station_cfg=stations,
            variables_to_keep=sel_vars
        )

    print("\n=== WRF Pipeline Completed ===")