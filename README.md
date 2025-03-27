# Efficient-3D-kernels-for-molecular-property-prediction 

## 📥 Dataset Download 
To run the model, you need to download the dataset using the [link](https://drive.google.com/drive/folders/1jW6Dz8wzTAipr_5852uM9yZObBn0S8OH?usp=sharing):


## Structure of files

|---- 3DGHK.py       
|---- load_data.py     
|---- datasets/ \
│    |---- .sdf files \
|---- README.md                    
|---- requirements.txt    

## 🔧 Dependencies
* Ensure you have the following dependencies installed before running the script:

## 🚀 Running the Model

` python 3DGHK.py --dataset gamma --kernel_type linear3dghk --seed 41 --num_split 10`\
\
 **Arguments**
- dataset : Name of the dataset to use (eg 1798, gamma etc.).
- kernel_type : Type of kernel (linear3dghk, cmg_kernel).
- seed : Random seed (41 for tox21 and 42 for others).
- num_split : Number of data splits.
## 📝 Citation

If you use this code in your research, please cite:  

@article{,\
  author    = {},\
  title     = {Efficient-3D-kernels-for-molecular-property-predictions},\
  journal   = {},\
  year      = {},\
}
