import argparse
import sys
import os
from .pp import execute_wrf_pipeline, execute_cmaq_pipeline
from .utils import timing

@timing
def main():
    parser = argparse.ArgumentParser(
        description="WRF and CMAQ NetCDF Post-Processing Tool."
    )
    parser.add_argument(
        "mode",
        choices=["wrf", "cmaq"],
        help="The processing mode."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the JSON configuration file."
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file not found at {args.config_path}")
        sys.exit(1)

    try:
        if args.mode == "wrf":
            execute_wrf_pipeline(args.config_path)
        elif args.mode == "cmaq":
            execute_cmaq_pipeline(args.config_path)
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()