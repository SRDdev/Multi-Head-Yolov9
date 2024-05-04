"""
Module for training YOLOv9 model.
"""
import argparse
import os
import subprocess
from roboflow import Roboflow

def install_packages():
    """Install required packages."""
    # Check if the git repository already exists
    if not os.path.exists("/content/yolov9"):
        subprocess.run("git clone https://github.com/WongKinYiu/yolov9", shell=True, check=True)

def get_data(api_key: str) -> str:
    """Download the dataset."""
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("srddev").project("deepfashion2-bfwof")
    version = project.version(1)
    dataset = version.download("yolov9")
    return dataset

def run_training(config_file_path: str) -> None:
    """Run the training process."""
    cmd = f"python /content/yolov9/segment/train.py --config-file {config_file_path}"
    subprocess.run(cmd, shell=True, check=True)

def main(config_file: str) -> None:
    """Main function."""
    install_packages()
    api_key = "api_key"  # Assuming userdata.get('roboflow') provides the API key
    get_data(api_key)
    run_training(config_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv9 with provided configuration file.")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
    