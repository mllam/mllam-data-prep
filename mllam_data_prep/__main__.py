from pathlib import Path

from .create_dataset import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file", type=Path)
    args = parser.parse_args()

    main(fp_config=args.config)
