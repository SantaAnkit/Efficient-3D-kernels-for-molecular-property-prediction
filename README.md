# Efficient-3D-kernels-for-molecular-property-prediction 

## 📥 Dataset Download 
To run the model, you need to download the dataset using the [link](https://drive.google.com/drive/folders/1jW6Dz8wzTAipr_5852uM9yZObBn0S8OH?usp=sharing):

## 📥 Supplymentary Material
The supplymentary material can be found [here](https://drive.google.com/file/d/11oOnCy-pxiRGAnfZZXyJiMVCMGgumKex/view?usp=drive_link).


## Structure of files
```
|---- 3DGHK.py       
|---- load_data.py     
|---- datasets/ \
│     |---- .sdf files \
|---- README.md                    
|---- requirements.txt    
```
## 🔧 Dependencies
* Ensure that dependencies are installed before running the script.

## 🚀 Running the Model

` python 3DGHK.py --dataset gamma --kernel_type linear3dghk --seed 41 --num_split 10`\
\
 **Arguments**
 ```
- dataset : Name of the dataset to use (eg 1798, gamma etc.).
- kernel_type : Type of kernel (linear3dghk, cmg_kernel).
- seed : Random seed (41 for tox21 and 42 for others).
- num_split : Number of data splits.
```
## 📝 Citation

If you use this code in your research, please cite:  

```bibtex
@article{10.1093/bioinformatics/btaf208,
    author = {Ankit and Bhadra, Sahely and Rousu, Juho},
    title = {Efficient 3D kernels for molecular property prediction},
    journal = {Bioinformatics},
    volume = {41},
    number = {Supplement_1},
    pages = {i58-i67},
    year = {2025},
    month = {07},
    abstract = {This paper addresses the challenge of incorporating 3-dimensional (3D) structural information in graph kernels for machine learning-based virtual screening, a crucial task in drug discovery. Existing kernels that capture 3D        information often suffer from high computational complexity, which limits their scalability.To overcome this, we propose the 3D chain motif graph kernel, which effectively integrates essential 3D structural properties—bond length, bond        angle, and torsion angle—within the three-hop neighborhood of each atom in a molecule. In addition, we introduce a more computationally efficient variant, the 3D graph hopper kernel (3DGHK), which reduces the complexity from the state-of-     the-art O(n6) (for the 3D pharmacophore kernel) to O(n2(m+log(n)+δ2+dT6)). Here, n is the number of nodes, T is the highest degree of the node, m is the number of edges, δ is the diameter of the graph, and d is the dimension of the            attributes of the nodes. We conducted experiments on 21 datasets, demonstrating that 3DGHK outperforms other state-of-the-art 2D and 3D graph kernels, but it also surpasses deep learning models in classification accuracy, offering a           powerful and scalable solution for virtual screening tasks.Our code is publicly available at https://github.com/SantaAnkit/Efficient-3D-kernels-for-molecular-property-prediction.git.},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf208},
    url = {https://doi.org/10.1093/bioinformatics/btaf208},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/41/Supplement\_1/i58/63745297/btaf208.pdf},
}
```
