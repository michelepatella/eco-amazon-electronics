import glob
import os


def get_latest_checkpoint(model_name):
    """Get the latest model checkpoint from its folder."""
    saved_model_dir = f"./saved/{model_name}"
    checkpoint_files = glob.glob(os.path.join(saved_model_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoint found at '{saved_model_dir}'.")
    latest_file = sorted(checkpoint_files)[-1]
    return latest_file
