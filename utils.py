import glob
import os


def get_latest_checkpoint(model_name):
    """Get the latest model checkpoint from its folder."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    saved_model_dir = os.path.join(base_path, "2_recbole", "models", model_name)
    checkpoint_files = glob.glob(os.path.join(saved_model_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoint found at '{saved_model_dir}'.")
    latest_file = sorted(checkpoint_files)[-1]
    return latest_file
