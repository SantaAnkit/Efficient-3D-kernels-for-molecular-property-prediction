import os
import torch
import math
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors

def mol2graph(mol, D=3):
    try:
        conf = mol.GetConformer()
    except Exception as e:
        smiles = AllChem.MolToSmiles(mol)
        print(f'smiles:{smiles} error message:{e}')

    atom_pos = []
    atomic_num_list = []
    all_atom_features = []

    # Get atom attributes and positions
    rdPartialCharges.ComputeGasteigerCharges(mol)

    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        atomic_num_list.append(atomic_num)
        atom_feature = get_atom_rep(atom)
        if D == 2:
            atom_pos.append(
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
        elif D == 3:
            atom_pos.append([conf.GetAtomPosition(
                i).x, conf.GetAtomPosition(i).y,
                             conf.GetAtomPosition(i).z])
        all_atom_features.append(atom_feature)
    # Add extra features that are needs to calculate using mol
    all_atom_features = get_extra_atom_feature(all_atom_features, mol)

    # Get bond attributes
    edge_list = []
    edge_attr_list = []
    for idx, bond in enumerate(mol.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_attr = []
        bond_attr += one_hot_vector(
            bond.GetBondTypeAsDouble(),
            [1.0, 1.5, 2.0, 3.0]
        )

        is_aromatic = bond.GetIsAromatic()
        is_conjugate = bond.GetIsConjugated()
        is_in_ring = bond.IsInRing()
        bond_attr.append(is_aromatic)
        bond_attr.append(is_conjugate)
        bond_attr.append(is_in_ring)

        edge_list.append((i, j))
        edge_attr_list.append(bond_attr)

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)

    x = torch.tensor(all_atom_features, dtype=torch.float32)
    p = torch.tensor(atom_pos, dtype=torch.float32)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    atomic_num = torch.tensor(atomic_num_list, dtype=torch.int)


    data = Data(x=x, p=p, edge_index=edge_index,
                edge_attr=edge_attr, atomic_num=atomic_num)  # , adj=adj,
    return data

def get_atom_rep(atom):
    features = []
    # H, C, N, O, F, Si, P, S, Cl, Br, I, other
    features += one_hot_vector(atom.GetAtomicNum(), [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 999])
    features += one_hot_vector(len(atom.GetNeighbors()), list(range(1, 5)))

    features.append(atom.GetFormalCharge())
    features.append(atom.IsInRing())
    features.append(atom.GetIsAromatic())
    features.append(atom.GetExplicitValence())
    features.append(atom.GetMass())

    # Add Gasteiger charge and set to 0 if it is NaN or Infinite
    gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
    if math.isnan(gasteiger_charge) or math.isinf(gasteiger_charge):
        gasteiger_charge = 0
    features.append(gasteiger_charge)

    # Add Gasteiger H charge and set to 0 if it is NaN or Infinite
    gasteiger_h_charge = float(atom.GetProp('_GasteigerHCharge'))
    if math.isnan(gasteiger_h_charge) or math.isinf(gasteiger_h_charge):
        gasteiger_h_charge = 0

    features.append(gasteiger_h_charge)
    return features
    
def one_hot_vector(val, lst):
	'''
	Converts a value to a one-hot vector based on options in lst
	'''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)
    
def get_extra_atom_feature(all_atom_features, mol):
    '''
    Get more atom features that cannot be calculated only with atom,
    but also with mol
    :param all_atom_features:
    :param mol:
    :return:
    '''
    # Crippen has two parts: first is logP, second is Molar Refactivity(MR)
    all_atom_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    all_atom_TPSA_contrib = rdMolDescriptors._CalcTPSAContribs(mol)
    all_atom_ASA_contrib = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    all_atom_EState = EState.EStateIndices(mol)

    new_all_atom_features = []
    for atom_id, feature in enumerate(all_atom_features):
        crippen_logP = all_atom_crippen[atom_id][0]
        crippen_MR = all_atom_crippen[atom_id][1]
        atom_TPSA_contrib = all_atom_TPSA_contrib[atom_id]
        atom_ASA_contrib = all_atom_ASA_contrib[atom_id]
        atom_EState = all_atom_EState[atom_id]

        feature.append(crippen_logP)
        feature.append(crippen_MR)
        feature.append(atom_TPSA_contrib)
        feature.append(atom_ASA_contrib)
        feature.append(atom_EState)

        new_all_atom_features.append(feature)
    return new_all_atom_features   
def process_dataset(dataset_name):
    """
    Process the dataset and return a list of molecule graphs.
    :param dataset_name: Name of the dataset (used to locate SDF files)
    :return: Active molecules and Inactive molecules
    """
    
    mol_data= ['1798','1843','2258','2689','435008', '435034','463087','485290']
    fol = f'moldata/datasets'
    active_sdf = os.path.join(fol, f'{dataset_name}_actives_new.sdf')
    inactive_sdf = os.path.join(fol, f'{dataset_name}_inactives_new.sdf')
    
    data_list = []
    invalid_id_list = []
    counter = -1
    
    for file_name, label in [(active_sdf, 0), (inactive_sdf, 1)]:
        sdf_supplier = Chem.SDMolSupplier(file_name)
        
        for i, mol in tqdm(enumerate(sdf_supplier), desc=f"Processing {file_name}"):
            counter += 1
            if mol is None:
                invalid_id_list.append([counter, label])
                continue
            
            data = mol2graph(mol)
            if data is None:
                invalid_id_list.append([counter, label])
                continue
            
            data.idx = counter
            data.y = torch.tensor([label], dtype=torch.int)
            data.smiles = Chem.MolToSmiles(mol)
            data_list.append(data)
            
    if dataset_name in mol_data:
        active_molecules = [data for data in data_list if data.y.item() == 0]
        inactive_molecules = [data for data in data_list if data.y.item() == 1]
      
    else:
        data_list = [data for data in data_list if len(data.edge_index) >1]
        active_molecules = [data for data in data_list if data.y.item() == 0]
        inactive_molecules = [data for data in data_list if data.y.item() == 1]
	
    return active_molecules,inactive_molecules
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process molecular dataset into graph format.")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., '1798 or AhR')")
    args = parser.parse_args()
    
    active_molecules,inactive_molecules = process_dataset(args.dataset_name)
    print(f"Processed {len(active_molecules)} active molecules.")
    print(f"Processed {len(inactive_molecules)} inactive molecules.")
