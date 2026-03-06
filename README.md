#  3D Motion Capture from Stereo Vision

## 📌 Overview
This project (part of the French "TIPE" curriculum) focuses on reconstructing 3D motion from synchronized video feeds of objects equipped with colored markers. 

The goal was to find the optimal balance between **metrological precision** and **execution speed (FPS)** for real-time Motion Capture applications.

## 🚀 Key Features
* **Rigorous Calibration:** Focal length estimation using linear regression, validated by **Monte Carlo methods** to quantify measurement uncertainty.
* **Optimized Image Processing:** Marker detection via HSV thresholding and vectorized center-of-mass calculations using `numpy`.
* **3D Triangulation:** Solving an overdetermined system using the Least Squares method ($A^T Ax = A^T b$) to reconstruct $(X, Y, Z)$ coordinates.
* **Performance Analysis:** Study of the influence of image resolution on inter-point distance precision and fluidity (achieving up to 120 FPS).

## 🛠 Project Structure
* `main.py`: Main script for processing and reconstruction.
* `monte_carlo_focale.py`: Calibration script for focal length estimation.
* `parametres_et_fonctions.py`: Core library (detection, transformations, 3D math).
* `traitement_videos.py`: Pre-processing scripts (video slicing and resizing).
* `analyse_resultats.py`: Generates performance and error graphs.

## 📊 Results
The system meets realism requirements with noise levels below 3% on trajectories, while maintaining high frame rates thanks to code vectorization.

## ⚙️ Installation
1. Clone the repo: `git clone https://github.com/your-name/tipe-mocap.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Place your images in the `images3/` folder (a small sample is provided in `sample_data/`).
