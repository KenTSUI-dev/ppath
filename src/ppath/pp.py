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

    # specific options for WRF to ensure smooth concatenation
    # combine='nested' and concat_dim='Time' are standard for WRF output
    try:
        ds = xr.open_mfdataset(
            sorted(file_paths),
            concat_dim='Time',
            combine='nested',
            parallel=False,
            chunks={'Time' : 1}
        )

        # Times is a 2D variable in NetCDF and is auto decoded as 1D time array by xarray when loaded.
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
        # CMAQ uses TSTEP as the unlimited time dimension
        ds = xr.open_mfdataset(
            sorted(file_paths),
            concat_dim='TSTEP',
            combine='nested',
            parallel=False,
            chunks={'TSTEP': 1}
        )

        # Sort by TFLAG if it exists
        if 'TFLAG' in ds:
            # TFLAG structure is typically: (TSTEP, VAR, DATE-TIME)
            # We only need the flags from the first variable (VAR=0) to determine the timestep order.
            tflag_ref = ds['TFLAG'].isel(VAR=0)

            # Extract Date (YYYYDDD) and Time (HHMMSS)
            # The dimension name 'DATE-TIME' has a hyphen, so we use dictionary unpacking
            # to pass it as a keyword argument to isel.
            # Index 0 = Date, Index 1 = Time
            dates = tflag_ref.isel(**{'DATE-TIME': 0})
            times = tflag_ref.isel(**{'DATE-TIME': 1})

            # Create a composite integer key for sorting: YYYYDDDHHMMSS
            # This is significantly faster than converting to datetime objects for sorting
            sort_key = (dates.astype('int64') * 1000000) + times.astype('int64')

            # Sort the dataset using this computed key
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
    Assigns units if provided in the configuration.

    Args:
        calc_config (list): List of dicts containing 'var', 'formula', and optional 'unit'.
        loaded_datasets (dict): Dictionary of loaded xarray Datasets keyed by label.

    Returns:
        dict: Dictionary of calculated xarray DataArrays.
    """
    calculated_vars = {}
    print("\n--- Calculating Variables ---")

    for calc_item in calc_config:
        var_name = calc_item.get("var")
        formula = calc_item.get("formula")
        unit = calc_item.get("unit")  # Get unit from config

        if not var_name or not formula:
            continue

        print(f"Processing: {var_name}")

        # Helper function for Regex substitution
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

            # Evaluate lazily (xarray/dask)
            result_da = eval(executable_formula, {"loaded_datasets": loaded_datasets, "np": np})

            # Assign metadata
            result_da.name = var_name
            if unit:
                result_da.attrs['units'] = unit

            calculated_vars[var_name] = result_da

        except Exception as e:
            print(f"Error calculating '{var_name}': {e}")

    return calculated_vars

def process_wrf_netcdf(input_source, output_path, variables_to_keep=None):
    """
    Converts WRF NetCDF file(s) into an ArcGIS Voxel Layer compliant NetCDF.

    Args:
        input_source (str, list, or xr.Dataset): Path to file(s), list of paths,
                                                 or an existing xarray Dataset.
        output_path (str): Destination path for the NetCDF file.
        variables_to_keep (list): List of variable names to process.
    """

    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_wrf_dataset(input_source)

    # --- 1. Generate 1D Spatial Coordinates (Degrees) ---
    # We need the dimensions of the unstaggered (mass) grid
    nx = ds.sizes['west_east']
    ny = ds.sizes['south_north']
    nz = ds.sizes['bottom_top']

    # Extract 2D Lat/Lon from the first time step to calculate average spacing
    # XLAT is (Time, south_north, west_east)
    try:
        lat_2d = ds['XLAT'].isel(Time=0).values
        lon_2d = ds['XLONG'].isel(Time=0).values
    except KeyError:
        print("Error: XLAT or XLONG not found. Cannot calculate degree coordinates.")
        return

    # Calculate average delta Y (Latitude) and delta X (Longitude)
    # Axis 0 is south_north, Axis 1 is west_east
    dy_deg = np.mean(np.diff(lat_2d, axis=0))
    dx_deg = np.mean(np.diff(lon_2d, axis=1))

    # Get origin (lower-left corner)
    y0 = lat_2d[0, 0]
    x0 = lon_2d[0, 0]

    # Create 1D arrays representing degrees
    x_coords = (x0 + np.arange(nx) * dx_deg).astype('float32')
    y_coords = (y0 + np.arange(ny) * dy_deg).astype('float32')
    z_coords = np.arange(nz, dtype='float32')

    # --- 2. Generate Time Coordinate (Float Hours) ---
    base_date = datetime(1900, 1, 1)

    if 'Times' in ds.variables:
        raw_times = ds['Times'].values
        time_values = []
        for t in raw_times:
            try:
                # Decode bytes to string (e.g., "2025-11-17_00:00:00")
                t_str = t.decode('utf-8')
                dt = datetime.strptime(t_str, '%Y-%m-%d_%H:%M:%S')
                # Convert to float hours immediately
                time_values.append(calculate_hours_since_base(dt, base_date))
            except Exception:
                # Fallback
                time_values.append(0.0)

        time_coords = np.array(time_values, dtype='float32')
    else:
        print("Warning: 'Times' variable not found. Creating dummy time index.")
        time_coords = np.arange(ds.sizes['Time'], dtype='float32')

    # --- 3. Create New Compliant Dataset ---
    new_ds = xr.Dataset()

    new_ds = new_ds.assign_coords({
        'time': time_coords,
        'level': z_coords,
        'y': y_coords,
        'x': x_coords
    })

    # Set CF Attributes for Coordinates
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

    # --- 4. Transfer and Destagger Data Variables ---
    # Map original mass-grid dimensions to new coordinate names
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

        # --- Destaggering Logic ---
        # Check for staggered dimensions and average them to the mass grid size

        # 1. Destagger West-East (U-grid)
        if 'west_east_stag' in dims:
            axis_idx = dims.index('west_east_stag')
            # Average adjacent values along the axis: 0.5 * (val[i] + val[i+1])
            sl_left = [slice(None)] * values.ndim
            sl_right = [slice(None)] * values.ndim
            sl_left[axis_idx] = slice(None, -1)
            sl_right[axis_idx] = slice(1, None)

            values = 0.5 * (values[tuple(sl_left)] + values[tuple(sl_right)])
            # Rename dimension in local list
            dims[axis_idx] = 'west_east'

        # 2. Destagger South-North (V-grid)
        if 'south_north_stag' in dims:
            axis_idx = dims.index('south_north_stag')
            sl_left = [slice(None)] * values.ndim
            sl_right = [slice(None)] * values.ndim
            sl_left[axis_idx] = slice(None, -1)
            sl_right[axis_idx] = slice(1, None)

            values = 0.5 * (values[tuple(sl_left)] + values[tuple(sl_right)])
            dims[axis_idx] = 'south_north'

        # 3. Destagger Bottom-Top (W-grid)
        if 'bottom_top_stag' in dims:
            axis_idx = dims.index('bottom_top_stag')
            sl_left = [slice(None)] * values.ndim
            sl_right = [slice(None)] * values.ndim
            sl_left[axis_idx] = slice(None, -1)
            sl_right[axis_idx] = slice(1, None)

            values = 0.5 * (values[tuple(sl_left)] + values[tuple(sl_right)])
            dims[axis_idx] = 'bottom_top'

        # --- Map to Output Dimensions ---
        new_dims = [dim_map.get(d, d) for d in dims]

        # Only add if dimensions match our new coordinate system
        if all(d in new_ds.coords for d in new_dims):
            print(f"  Converting variable: {var_name} (Destaggered if necessary)")
            new_ds[var_name] = (new_dims, values)

            attrs = src_var.attrs.copy()
            # Clean up attributes that might confuse ArcGIS
            if 'coordinates' in attrs: del attrs['coordinates']
            if 'stagger' in attrs: del attrs['stagger']

            new_ds[var_name].attrs = attrs
            new_ds[var_name].attrs['_FillValue'] = -999.0

    # --- 5. Write Output with Specific Encoding ---
    print(f"Writing to {output_path}...")


    new_ds.to_netcdf(
        output_path,
        format='NETCDF4',
    )
    print("Done.")


def process_cmaq_netcdf(input_source, output_path, variables_to_keep=None):
    """
    Converts CMAQ (ACONC/APMDIAG) NetCDF file(s) or an existing xarray Dataset
    into an ArcGIS Voxel Layer compliant NetCDF.

    Args:
        input_source (str, list, or xr.Dataset): Path to file(s), list of paths,
                                                 or an existing xarray Dataset.
        output_path (str): Destination path for the NetCDF file.
        variables_to_keep (list): List of variable names to process.
    """

    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_cmaq_dataset(input_source)

    # --- 1. Generate 1D Spatial Coordinates (Lat/Lon) ---
    # Read IOAPI Projection Parameters
    # GDTYP=2 is Lambert Conformal Conic, which is standard for CMAQ
    try:
        p_alp = ds.attrs.get('P_ALP')  # Standard Parallel 1
        p_bet = ds.attrs.get('P_BET')  # Standard Parallel 2
        p_gam = ds.attrs.get('P_GAM')  # Central Meridian
        y_cent = ds.attrs.get('YCENT')  # Latitude of origin
        # XCENT is often the longitude origin, but in PROJ4 +lon_0 is usually P_GAM for LCC

        x_orig = ds.attrs.get('XORIG', 0.0)
        y_orig = ds.attrs.get('YORIG', 0.0)
        x_cell = ds.attrs.get('XCELL', 1000.0)
        y_cell = ds.attrs.get('YCELL', 1000.0)

        ncol = ds.sizes['COL']
        nrow = ds.sizes['ROW']
        nlay = ds.sizes['LAY']

        # Define Projection (CMAQ usually uses a spherical earth radius of 6370000m)
        proj_str = (f"+proj=lcc +lat_1={p_alp} +lat_2={p_bet} +lat_0={y_cent} "
                    f"+lon_0={p_gam} +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs")

        p = pyproj.Proj(proj_str)

        # Create 1D arrays in Meters (Projected)
        x_m = x_orig + (np.arange(ncol) * x_cell)
        y_m = y_orig + (np.arange(nrow) * y_cell)

        # Create 2D Meshgrid in Meters
        xx_m, yy_m = np.meshgrid(x_m, y_m)

        # Unproject to Longitude/Latitude (2D)
        lon_2d, lat_2d = p(xx_m, yy_m, inverse=True)

        # Calculate average delta (Degrees)
        dy_deg = np.mean(np.diff(lat_2d, axis=0))
        dx_deg = np.mean(np.diff(lon_2d, axis=1))

        # Calculate Origin (Lower Left)
        lat0 = lat_2d[0, 0]
        lon0 = lon_2d[0, 0]

        # Generate 1D Degree Coordinates
        x_coords = (lon0 + np.arange(ncol) * dx_deg).astype('float32')
        y_coords = (lat0 + np.arange(nrow) * dy_deg).astype('float32')
        z_coords = np.arange(nlay, dtype='float32')

    except Exception as e:
        print(f"Error calculating Lat/Lon from projection: {e}")
        print("Falling back to generic index coordinates (this may not align spatially).")
        x_coords = np.arange(ds.sizes['COL'], dtype='float32')
        y_coords = np.arange(ds.sizes['ROW'], dtype='float32')
        z_coords = np.arange(ds.sizes['LAY'], dtype='float32')

    # --- 2. Generate Time Coordinate (Float Hours) ---
    base_date = datetime(1900, 1, 1)

    if 'TFLAG' in ds.variables:
        tflag = ds['TFLAG'].values
        dates = tflag[:, 0, 0]  # YYYYDDD
        times = tflag[:, 0, 1]  # HHMMSS

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
        print("Warning: 'TFLAG' variable not found. Creating dummy time index.")
        time_coords = np.arange(ds.sizes['TSTEP'], dtype='float32')

    # --- 3. Create New Compliant Dataset ---
    new_ds = xr.Dataset()

    new_ds = new_ds.assign_coords({
        'time': time_coords,
        'level': z_coords,
        'y': y_coords,
        'x': x_coords
    })

    # Set CF Attributes (Degrees)
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

    # --- 4. Transfer Data Variables ---
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

    # --- 5. Write Output with Specific Encoding ---
    print(f"Writing to {output_path}...")


    new_ds.to_netcdf(
        output_path,
        format='NETCDF4',
    )
    print("Done.")


def extract_wrf_timeseries(input_source, output_path, station_cfg, variables_to_keep=None):
    """
    Extracts time-series data from WRF NetCDF files (or Dataset) for specific grid cells
    (stations) and saves the result to a CSV file.

    Includes automatic destaggering for U, V, and W grid variables.

    Args:
        input_source (str, list, or xr.Dataset): Path to file(s), list of paths,
                                                 or an existing xarray Dataset.
        output_path (str): Destination path for the CSV file.
        station_cfg (list): List of dictionaries containing station info.
                            Expected format: [{"label": "Name", "west_east": int, "south_north": int, "bottom_top": int}, ...]
        variables_to_keep (list): List of variable names to process.
    """

    # 1. Load Dataset if input is not already an xarray Dataset
    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_wrf_dataset(input_source)

    print(f"\nProcessing Time Series Output: {output_path}")

    # 2. Validate Variables
    if variables_to_keep:
        valid_vars = [v for v in variables_to_keep if v in ds.variables]
        missing_vars = [v for v in variables_to_keep if v not in ds.variables]
        if missing_vars:
            print(f"Warning: The following requested variables were not found: {missing_vars}")
    else:
        exclude = ['Times', 'XLAT', 'XLONG', 'XTIME', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V']
        valid_vars = [v for v in ds.data_vars if v not in exclude]

    # 3. Pre-process Time Variable
    # We calculate the timeline ONCE for the whole dataset.
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

    # Fallback to numeric time if parsing failed
    if dt_index is None and 'XTIME' in ds:
        dt_index = ds['XTIME'].values

    # --- 4. Lazy Destaggering Logic ---
    processed_vars = {}

    # Mapping: Staggered Dimension -> Unstaggered (Mass) Dimension
    stag_map = {
        'west_east_stag': 'west_east',
        'south_north_stag': 'south_north',
        'bottom_top_stag': 'bottom_top'
    }

    print("Preparing variables (Lazy Destaggering)...")

    for var_name in valid_vars:
        da = ds[var_name]
        dims = da.dims

        # Check if variable has any staggered dimensions
        for stag_dim, mass_dim in stag_map.items():
            if stag_dim in dims:
                # Destaggering formula: 0.5 * (val[i] + val[i+1])
                # We use lazy slicing. Dask will only compute the slices needed for the
                # specific stations requested later, not the whole grid.
                da = 0.5 * (da.isel({stag_dim: slice(0, -1)}) +
                            da.isel({stag_dim: slice(1, None)}))

                # Rename dimension to match mass grid for uniform indexing
                da = da.rename({stag_dim: mass_dim})

        processed_vars[var_name] = da

    # Create a temporary working dataset.
    # We do NOT include coordinates here to prevent index alignment issues during selection.
    working_ds = xr.Dataset(processed_vars)

    # --- 5. Vectorized Index Preparation ---
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

        # Bounds Check (against mass grid dimensions)
        if (we < ds.sizes['west_east'] and
                sn < ds.sizes['south_north'] and
                bt < ds.sizes['bottom_top']):

            we_idxs.append(we)
            sn_idxs.append(sn)
            bt_idxs.append(bt)
            labels.append(lbl)
            valid_stations.append(stn)
        else:
            print(f"Skipping {lbl}: Indices (WE:{we}, SN:{sn}, BT:{bt}) out of bounds.")

    if not valid_stations:
        print(f"No valid stations found for {output_path}")
        return

    # Create DataArrays for Advanced Indexing
    # This creates a new dimension 'station' in the result
    idx_station = xr.DataArray(np.arange(len(valid_stations)), dims="station")
    idx_we = xr.DataArray(we_idxs, dims="station")
    idx_sn = xr.DataArray(sn_idxs, dims="station")
    idx_bt = xr.DataArray(bt_idxs, dims="station")

    # --- 6. Extract and Compute  ---
    try:
        print(f"Extracting data for {len(valid_stations)} stations...")

        # Vectorized selection:
        # This selects all stations at all time steps in one symbolic operation.
        subset = working_ds.isel(
            west_east=idx_we,
            south_north=idx_sn,
            bottom_top=idx_bt
        )

        # TRIGGER COMPUTE:
        # This is the single heavy I/O operation. It reads only the required chunks.
        subset_loaded = subset.compute()

        # --- 7. Convert to DataFrame and Format ---
        # Convert to DataFrame (fast, as data is now in memory)
        df = subset_loaded.to_dataframe()

        # Reset index to make 'station' and 'Time' accessible as columns
        df = df.reset_index()

        # Create Metadata DataFrame to map 'station' index back to Label/Coords
        meta_df = pd.DataFrame({
            'station': np.arange(len(valid_stations)),
            'Station': labels,
            'west_east': we_idxs,
            'south_north': sn_idxs,
            'bottom_top': bt_idxs
        })

        # Merge metadata
        final_df = pd.merge(df, meta_df, on='station', how='left')

        # Handle DateTime Mapping
        # We assume the dataset is ordered by 'Time' (integer index)
        if dt_index is not None:
            # Check if we have 'Time' column (standard WRF)
            if 'Time' in final_df.columns:
                # Ensure we don't go out of bounds if dt_index is shorter than dataset (rare)
                max_t = final_df['Time'].max()
                if max_t < len(dt_index):
                    # Map integer Time index to actual DateTime
                    # We use a lookup array for speed
                    final_df['DateTime'] = dt_index[final_df['Time'].values]
                else:
                    print("Warning: Time index mismatch. Using raw Time index.")
                    final_df['DateTime'] = final_df['Time']
            else:
                # Fallback if 'Time' dim was dropped or named differently
                final_df['DateTime'] = final_df.index

        # Organize Columns
        meta_cols = ['Station', 'DateTime', 'west_east', 'south_north', 'bottom_top']
        data_cols = [c for c in valid_vars if c in final_df.columns]

        # Ensure columns exist
        avail_meta = [c for c in meta_cols if c in final_df.columns]
        final_cols = avail_meta + data_cols

        final_df = final_df[final_cols]
        final_df = final_df.sort_values(by=avail_meta)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Saved time series to {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()


def extract_cmaq_timeseries(input_source, output_path, station_cfg, variables_to_keep=None):
    """
    Extracts time-series data from CMAQ NetCDF files (or Dataset) for specific grid cells
    (stations) and saves the result to a CSV file.

    Args:
        input_source (str, list, or xr.Dataset): Path to file(s), list of paths,
                                                 or an existing xarray Dataset.
        output_path (str): Destination path for the CSV file.
        station_cfg (list): List of dictionaries containing station info.
                            Expected format: [{"label": "Name", "row": int, "col": int, "lay": int}, ...]
        variables_to_keep (list): List of variable names to process.
    """

    # 1. Load Dataset if input is not already an xarray Dataset
    if isinstance(input_source, xr.Dataset):
        ds = input_source
    else:
        ds = load_cmaq_dataset(input_source)

    print(f"\nProcessing Time Series Output: {output_path}")

    # 2. Validate Variables
    # Ensure we don't try to extract TFLAG via the spatial extractor, handle it separately
    if variables_to_keep:
        valid_vars = [v for v in variables_to_keep if v in ds.variables]
        missing = [v for v in variables_to_keep if v not in ds.variables]
        if missing:
            print(f"Warning: Missing variables: {missing}")
    else:
        valid_vars = [v for v in ds.data_vars if v != 'TFLAG']

    # 3. Pre-calculate DateTimes (Vectorized)
    # We calculate the timeline ONCE for the whole dataset, not per station.
    dt_index = None

    # Try TFLAG first (Standard CMAQ)
    if 'TFLAG' in ds:
        try:
            # Extract TFLAG. It is usually (TSTEP, VAR, DATE-TIME).
            # We load it immediately as it is small.
            tflag = ds['TFLAG'].isel(VAR=0).values  # Shape: (TSTEP, 2) [Date, Time]

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

    # Fallback to 'time' coordinate if TFLAG failed or missing
    if dt_index is None and 'time' in ds.coords:
        # If time is float hours (from previous processing), this might need adjustment,
        # but usually, if it's a datetime64, this works.
        dt_index = ds['time'].values

    # 4. Prepare Vectorized Indices
    # We create DataArrays for the indices to trigger Xarray's advanced indexing.
    # This allows us to select (row[0], col[0], lay[0]), (row[1], col[1], lay[1]), etc.

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

        # Bounds check
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

    # Create Indexers with a new dimension 'station'
    # This tells xarray: "For the 'station' dimension, use these specific R, C, L coordinates"
    idx_station = xr.DataArray(np.arange(len(valid_stations)), dims="station")
    idx_row = xr.DataArray(rows, dims="station")
    idx_col = xr.DataArray(cols, dims="station")
    idx_lay = xr.DataArray(lays, dims="station")

    # 5. Extract Data (Lazy -> Compute)
    try:
        print(f"Extracting data for {len(valid_stations)} stations...")

        # Select the subset of variables
        # isel(ROW=..., COL=..., LAY=...) extracts the diagonal elements defined by the indices
        # Resulting dims: (TSTEP, station)
        subset = ds[valid_vars].isel(ROW=idx_row, COL=idx_col, LAY=idx_lay)

        # TRIGGER COMPUTE: This is the only heavy IO operation.
        # It loads the time series for all stations into memory at once.
        subset_loaded = subset.compute()

        # 6. Convert to DataFrame
        # Since subset_loaded is in memory, this is instant.
        # The dataframe will have a MultiIndex or index: (TSTEP, station)
        df = subset_loaded.to_dataframe()

        # Reset index to make TSTEP and station columns available
        df = df.reset_index()

        # 7. Post-Processing DataFrame

        # Map the integer 'station' index back to the Label and original coordinates
        # Create a lookup dictionary or dataframe
        stn_meta = pd.DataFrame({
            'station': np.arange(len(valid_stations)),
            'Station': labels,
            'Row': rows,
            'Col': cols,
            'Layer': lays
        })

        # Merge metadata into the main dataframe
        final_df = pd.merge(df, stn_meta, on='station', how='left')

        # Add DateTime column
        # We assume the dataframe is ordered by TSTEP.
        # If TSTEP is an index, we map it.
        if dt_index is not None:
            # Create a mapping from TSTEP index to Date
            # Assuming TSTEP in df corresponds to 0, 1, 2... indices of dt_index
            # If TSTEP is the actual value from NetCDF (0, 10000, etc), we might need rank.
            # Safest way: If 'TSTEP' is in columns, use rank. If 'time' is in columns, use it.

            if len(dt_index) == ds.sizes['TSTEP']:
                # Create a temporary DF to map TSTEP index to DateTime
                # We need to know how many timesteps per station
                n_timesteps = len(dt_index)
                n_stations = len(valid_stations)

                # If the reset_index gave us 'TSTEP' (integer) or 'time'
                if 'TSTEP' in final_df.columns:
                    # Often TSTEP is just an enumeration or a large integer.
                    # We rely on the sort order.
                    final_df = final_df.sort_values(by=['station', 'TSTEP'])

                    # Tile the dates for every station
                    # Ensure the length matches
                    if len(final_df) == n_timesteps * n_stations:
                        final_df['DateTime'] = np.tile(dt_index, n_stations)
                    else:
                        # Fallback if lengths don't align (e.g. NaNs dropped)
                        print("Warning: Data length mismatch. DateTime alignment might be off.")
                        final_df['DateTime'] = final_df['TSTEP']
                elif 'time' in final_df.columns:
                    # If we have the time coord, use it, but prefer the calculated dt_index if 'time' is float
                    final_df['DateTime'] = final_df['time']
            else:
                final_df['DateTime'] = final_df.index

        # Organize Columns
        meta_cols = ['Station', 'DateTime', 'Layer', 'Row', 'Col']
        data_cols = [c for c in valid_vars if c in final_df.columns]

        # Ensure columns exist
        avail_meta = [c for c in meta_cols if c in final_df.columns]
        final_cols = avail_meta + data_cols

        final_df = final_df[final_cols]
        final_df = final_df.sort_values(by=avail_meta)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Saved time series to {output_path}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

def execute_cmaq_pipeline(json_input):
    """
    Orchestrates the CMAQ processing pipeline: loading, calculation, merging, and output.

    Args:
        json_input (str, dict): JSON string, dictionary, or file path containing the configuration.
    """
    # 1. Parse JSON Input
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

    # 2. Load Datasets
    for source in config.get("source", []):
        label = source["label"]
        files = source["input_nc"]
        print(f"Loading source: {label}")

        # Load lazily using existing function
        ds = load_cmaq_dataset(files)

        # Capture TFLAG and Attributes from the first dataset to use for output later.
        # We need this because we are about to drop TFLAG/VAR to ensure successful merging.
        if reference_tflag is None and 'TFLAG' in ds:
            # Keep TFLAG but slice VAR dimension to size 1.
            # This prevents dimension mismatch errors later if we re-attach it.
            reference_tflag = ds['TFLAG'].isel(VAR=slice(0, 1))
            reference_attrs = ds.attrs

        # CRITICAL: Drop 'VAR' dimension and 'TFLAG' variable.
        # ACONC usually has VAR=226, APMDIAG has VAR=39.
        # xarray cannot merge datasets if shared dimensions (VAR) have different lengths.
        if 'TFLAG' in ds:
            ds = ds.drop_vars('TFLAG')
        if 'VAR' in ds.dims:
            ds = ds.drop_dims('VAR')

        loaded_datasets[label] = ds

    # 3. Check for Naming Conflicts
    existing_vars = set()
    for ds in loaded_datasets.values():
        existing_vars.update(ds.data_vars)

    for calc in config.get("calc_var", []):
        if calc["var"] in existing_vars:
            raise ValueError(
                f"Calculated variable name '{calc['var']}' conflicts with an existing variable in loaded datasets.")

    # 4. Calculate New Variables
    # This calls the existing function and returns a dictionary of DataArrays
    calculated_vars_dict = calculate_new_variables(config.get("calc_var", []), loaded_datasets)

    # Convert calculated dictionary to a Dataset for merging
    calc_ds = xr.Dataset(calculated_vars_dict)

    # 5. Merge Everything into "Dataset A"
    # We merge the loaded datasets and the calculated variables.
    datasets_to_merge = list(loaded_datasets.values()) + [calc_ds]

    print("\nMerging datasets...")
    dataset_a = xr.merge(datasets_to_merge, compat='override')

    # Restore Global Attributes (useful for projection info in process_cmaq_netcdf)
    if reference_attrs:
        dataset_a.attrs = reference_attrs

    # 6. Handle Voxel Output
    for v_out in config.get("voxel_ouput", []):
        out_path = v_out["output_nc"]
        sel_vars = v_out["selected_var"]

        print(f"\nProcessing Voxel Output: {out_path}")

        # RE-INJECT TFLAG
        # The process_cmaq_netcdf function relies on 'TFLAG' to calculate
        # the 'time' coordinate (hours since 1900). We attach the reference TFLAG we saved earlier.
        if reference_tflag is not None:
            dataset_a['TFLAG'] = reference_tflag

        # Call existing function
        process_cmaq_netcdf(dataset_a, out_path, variables_to_keep=sel_vars)

    # 7. Handle Time Series Output
    for ts_out in config.get("ts_ouput", []):
        out_csv = ts_out["output_csv"]
        sel_vars = ts_out["selected_var"]
        stations = ts_out["station"]

        # RE-INJECT TFLAG (Critical Step)
        # The extract_cmaq_timeseries function needs TFLAG to generate the DateTime column.
        # Since dataset_a had TFLAG dropped during merging, we must re-attach it here.
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
    Orchestrates the WRF processing pipeline: loading, calculation, merging,
    and outputting Voxel Layers (NetCDF) and Time Series (CSV).

    Args:
        json_input (str, dict): JSON string, dictionary, or file path containing the configuration.
    """
    # 1. Parse JSON Input
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

    # 2. Load Datasets
    # Iterate through sources and load them using the existing helper function
    for source in config.get("source", []):
        label = source["label"]
        files = source["input_nc"]
        print(f"Loading source: {label}")

        # Call existing function to load files lazily
        ds = load_wrf_dataset(files)
        loaded_datasets[label] = ds

    # 3. Check for Naming Conflicts
    # Ensure calculated variables do not overwrite existing variables in the source
    existing_vars = set()
    for ds in loaded_datasets.values():
        existing_vars.update(ds.data_vars)

    for calc in config.get("calc_var", []):
        if calc["var"] in existing_vars:
            raise ValueError(
                f"Calculated variable name '{calc['var']}' conflicts with an existing variable in loaded datasets."
            )

    # 4. Calculate New Variables
    calculated_vars_dict = calculate_new_variables(config.get("calc_var", []), loaded_datasets)

    # Convert calculated dictionary to a Dataset for merging
    calc_ds = xr.Dataset(calculated_vars_dict)

    # 5. Merge Everything into "Dataset A"
    # Combine loaded WRF data and calculated variables.
    print("\nMerging datasets...")
    datasets_to_merge = list(loaded_datasets.values()) + [calc_ds]
    dataset_a = xr.merge(datasets_to_merge, compat='override')

    # 6. Handle Voxel Output
    # Note: Using key 'voxel_ouput' as per the prompt's JSON structure
    for v_out in config.get("voxel_ouput", []):
        out_path = v_out["output_nc"]
        sel_vars = v_out["selected_var"]

        print(f"\nProcessing Voxel Output: {out_path}")

        # Create a subset with only selected variables (Lazy)
        # We check if variables exist to avoid KeyErrors
        valid_vars = [v for v in sel_vars if v in dataset_a]
        missing_vars = [v for v in sel_vars if v not in dataset_a]

        if missing_vars:
            print(f"Warning: The following variables were not found and skipped: {missing_vars}")

        # Call existing function to process and write NetCDF
        process_wrf_netcdf(dataset_a, out_path, variables_to_keep=valid_vars)

    # 7. Handle Time Series Output
    # Note: Using key 'ts_ouput' as per the prompt's JSON structure
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

