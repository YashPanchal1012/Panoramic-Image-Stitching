# Panoramic Image Stitching

This project builds panoramas in `hw1.ipynb` using OpenCV, NumPy, and Matplotlib.

## Install

From this directory, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install jupyter opencv-python numpy matplotlib
```

## Generate the panoramas

1. Keep the provided input photos in `Images/` (`Set1_*`, `Set2_*`, `exposure_*` and `cylindrical_*` images).
2. Start Jupyter from this project directory:

   ```bash
   jupyter notebook hw1.ipynb
   ```

3. In the notebook, choose **Kernel → Restart & Run All**. The notebook processes the image sets and saves panorama results and other visualizations in `outputs/`.
