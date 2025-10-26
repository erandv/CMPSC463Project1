This project performs unsupervised clustering of time-series segments from the PulseDB dataset using divide-and-conquer algorithms.  
It includes:
- Recursive clustering based on correlation similarity  
- Closest-pair detection within each cluster  
- Maximum subarray detection (Kadane’s Algorithm) to find active intervals  

Files:
- main.py — main program containing all functions  
- /out/ — folder where generated plots and CSV outputs are saved  

Dataset
Dataset used: VitalDB_AAMI_Test_Subset.mat
(from the PulseDB Balanced Training and Testing dataset on Kaggle)

Requirements:
Before running the program, make sure you install these libraries:

pip install numpy 
pip install matplotlib 
pip install h5py
