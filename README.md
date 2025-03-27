# Efficient-3D-kernels-for-molecular-property-prediction ## 

# 📥 Dataset Download #
To run the model, you need to download the dataset from the following link:

After downloading, place the dataset inside the datasets/ directory.
# 🔧 Dependencies
* Ensure you have the following dependencies installed before running the script:

# 🚀 Running the Model
 python 3DGHK.py --dataset gamma --kernel_type cmg_kernel --seed 41 --num_split 10
 
*--dataset : Name of the dataset to use.
*--kernel_type : Type of kernel (linear3d, cmg_kernel).
*--seed : Random seed (41 for tox21 and 42 for others).
*--num_split : Number of data splits.

├── 3DGHK.py                # Main script for training and evaluation
├── load_data.py            # Data loading and processing functions
├── dataset/
│   ├── .sdf files
├── README.md              
└── requirements.txt       


# 📝 Citation

If you use this code in your research, please cite:
@article{,
  author    = {},
  title     = {Efficient-3D-kernels-for-molecular-property-predictions},
  journal   = {},
  year      = {},
}
