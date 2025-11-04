# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:03:47 2025

@author: pc
"""

"""
分子指纹特征化模块
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MolecularFeaturizer:
    """分子指纹特征化类"""
    
    def __init__(self):
        self.solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
    
    def smiles_to_mol(self, smiles_list):
        """SMILES转分子对象"""
        mols = []
        valid_indices = []
        
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(idx)
            else:
                print(f"[Warning] Invalid SMILES at index {idx}: {smi}")
        
        return mols, valid_indices
    
    def get_maccs_fp(self, mol):
        """生成MACCS指纹"""
        fp = MACCSkeys.GenMACCSKeys(mol)
        return list(fp)
    
    def get_morgan_fp(self, mol, radius=2, nBits=1024):
        """生成Morgan指纹"""
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return list(fp)
    
    def get_rdkit_fp(self, mol, nBits=1024):
        """生成RDKit指纹"""
        fp = RDKFingerprint(mol, fpSize=nBits)
        return list(fp)
    
    def compute_fingerprints(self, smiles_list):
        """计算所有指纹"""
        print("Converting SMILES to molecules...")
        mols, valid_indices = self.smiles_to_mol(smiles_list)
        
        print("Computing fingerprints...")
        maccs_fps, morgan_fps, rdkit_fps = [], [], []
        
        for mol in tqdm(mols, desc="Fingerprints"):
            maccs_fps.append(self.get_maccs_fp(mol))
            morgan_fps.append(self.get_morgan_fp(mol))
            rdkit_fps.append(self.get_rdkit_fp(mol))
        
        return maccs_fps, morgan_fps, rdkit_fps, valid_indices
    
    def create_feature_dataframe(self, smiles_list, target_data, target_type='ABS'):
        """
        创建特征数据框
        
        Parameters:
        - smiles_list: SMILES列表
        - target_data: 包含目标值和溶剂信息的DataFrame
        - target_type: 目标类型 ('ABS', 'EM', 'LOG')
        """
        maccs_fps, morgan_fps, rdkit_fps, valid_indices = self.compute_fingerprints(smiles_list)
        
        # 创建指纹DataFrame
        df_maccs = pd.DataFrame(maccs_fps, columns=[f"MACCS_{i}" for i in range(166)])
        df_morgan = pd.DataFrame(morgan_fps, columns=[f"Morgan_{i}" for i in range(1024)])
        df_rdkit = pd.DataFrame(rdkit_fps, columns=[f"RDKit_{i}" for i in range(1024)])
        
        # 合并特征
        fps_combined = pd.concat([df_maccs, df_morgan, df_rdkit], axis=1)
        
        # 添加ID、溶剂信息和目标值
        base_data = target_data.iloc[valid_indices].reset_index(drop=True)
        final_df = pd.concat([base_data[['ID', 'SMILES', 'SOLVENT'] + self.solvent_cols], 
                             base_data[target_type], 
                             fps_combined], axis=1)
        
        print(f"Feature generation completed. Final dataset shape: {final_df.shape}")
        return final_df

def main():
    """主函数示例"""
    # 读取数据
    df = pd.read_excel('data/raw/absorption.xlsx')
    
    # 特征化
    featurizer = MolecularFeaturizer()
    feature_df = featurizer.create_feature_dataframe(
        df['SMILES'].tolist(), 
        df, 
        target_type='ABS'
    )
    
    # 保存结果
    feature_df.to_excel('data/processed/absorption_features.xlsx', index=False)

if __name__ == "__main__":
    main()