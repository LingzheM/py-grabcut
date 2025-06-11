A minimal, self‑contained re‑implementation of the GrabCut algorithm that avoids
OpenCV's built‑in `cv2.grabCut`.  It relies only on

* NumPy – array maths
* scikit‑learn – GaussianMixture for colour modelling
* PyMaxflow – graph‑cut / max‑flow optimisation
* scikit‑image – I/O convenience (can be replaced by imageio/Pillow)

Quick start  
$ pip install numpy scikit‑learn PyMaxflow scikit‑image  
$ python simple_grabcut.py input.jpg output.png --rect 50 50 200 300 --iter 5

The script will write an RGBA PNG whose alpha channel comes from the estimated
foreground mask.