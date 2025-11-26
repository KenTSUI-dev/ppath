from ppath.pp import execute_cmaq_pipeline, execute_wrf_pipeline

if __name__ == "__main__":
    execute_cmaq_pipeline("..\examples\cmaq_cfg.json")
    # execute_wrf_pipeline("..\examples\wrf_cfg.json")
    # execute_cmaq_pipeline("..\examples\cmaq_cfg_20251124.json")
    # execute_wrf_pipeline("..\examples\wrf_cfg_20251124.json")