# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 19:22:26 2025

@author: pc
"""


"""
特征选择模块
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm

class FeatureSelector:
    """特征选择器"""
    
    def __init__(self):
        self.id_cols = ['ID', 'SMILES']
        self.solvent_cols = ['SOLVENT', 'Et(30)', 'SP', 'SdP', 'SA', 'SB']
    
    def get_fingerprint_columns(self, df):
        """获取指纹特征列"""
        return [col for col in df.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    
    def variance_selection(self, df, target_col='ABS', threshold=0.01):
        """方差选择"""
        print("Performing variance selection...")
        fps_cols = self.get_fingerprint_columns(df)
        X_fp = df[fps_cols].copy()
        
        var_selector = VarianceThreshold(threshold=threshold)
        X_fp_var = var_selector.fit_transform(X_fp)
        retained_indices = var_selector.get_support(indices=True)
        retained_features = X_fp.columns[retained_indices].tolist()
        
        result_df = pd.concat([
            df[self.id_cols + self.solvent_cols],
            df[target_col],
            df[retained_features]
        ], axis=1)
        
        print(f"Variance selection: {len(retained_features)} features retained")
        return result_df, retained_features
    
    def correlation_selection(self, df, target_col='ABS', threshold=0.15):
        """相关性选择"""
        print("Performing correlation selection...")
        fps_cols = self.get_fingerprint_columns(df)
        X_fp = df[fps_cols].copy()
        y = df[target_col]
        
        corr_values = np.abs([np.corrcoef(X_fp[feat], y)[0, 1] for feat in fps_cols])
        retained_features = [feat for feat, corr in zip(fps_cols, corr_values) if corr > threshold]
        
        result_df = pd.concat([
            df[self.id_cols + self.solvent_cols],
            df[target_col],
            df[retained_features]
        ], axis=1)
        
        print(f"Correlation selection: {len(retained_features)} features retained")
        return result_df, retained_features
    
    def mutual_info_selection(self, df, target_col='ABS', threshold=0.01):
        """互信息选择"""
        print("Performing mutual information selection...")
        fps_cols = self.get_fingerprint_columns(df)
        X_fp = df[fps_cols].copy()
        y = df[target_col]
        
        mi_scores = mutual_info_regression(X_fp, y, random_state=42)
        retained_features = [feat for feat, score in zip(fps_cols, mi_scores) if score > threshold]
        
        # 保存MI得分
        mi_df = pd.DataFrame({
            "Feature": fps_cols,
            "MI_Score": mi_scores
        }).sort_values("MI_Score", ascending=False)
        
        result_df = pd.concat([
            df[self.id_cols + self.solvent_cols],
            df[target_col],
            df[retained_features]
        ], axis=1)
        
        print(f"Mutual information selection: {len(retained_features)} features retained")
        return result_df, retained_features, mi_df
    
    def recursive_feature_elimination(self, df, target_col='ABS', n_runs=100, model_type='LGB'):
        """递归特征消除"""
        print(f"Performing RFECV with {model_type} ({n_runs} runs)...")
        
        fps_cols = self.get_fingerprint_columns(df)
        X = df[fps_cols]
        y = df[target_col]
        
        # 创建分层折
        k = 10
        y_max, y_min = y.max(), y.min()
        y_gap = (y_max - y_min) / k
        y_hash = [int((val - y_min) // y_gap) for val in y]
        
        result_rank = pd.DataFrame(index=fps_cols)
        result_cv_score = pd.DataFrame(index=[f'feat_select_{i}' for i in range(1, len(fps_cols)+1)])
        
        for i in tqdm(range(n_runs), desc="RFECV runs"):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=i)
            
            if model_type == 'LGB':
                estimator = lgb.LGBMRegressor(
                    objective='regression', learning_rate=0.4, max_depth=4, 
                    num_leaves=6, feature_fraction=0.7, random_state=100
                )
            elif model_type == 'XGB':
                estimator = xgb.XGBRegressor(
                    learning_rate=0.2, max_depth=3, gamma=0.01, random_state=100
                )
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=100)
            
            rfecv = RFECV(estimator, cv=skf.split(X, y_hash), scoring='neg_root_mean_squared_error')
            rfecv.fit(X, y)
            
            result_rank[f'run_{i:02d}'] = rfecv.ranking_
            result_cv_score[f'run_{i:02d}'] = -rfecv.grid_scores_
        
        # 选择在至少100次的运行中被保留的特征
        rank_sum = np.sum(result_rank == 1, axis=1)
        threshold = rank_sum.values >= 100  #70,80,90,100
        selected_features = rank_sum.index[threshold].tolist()
        
        result_df = pd.concat([
            df[self.id_cols + self.solvent_cols],
            df[target_col],
            df[selected_features]
        ], axis=1)
        
        print(f"RFECV selection: {len(selected_features)} features retained")
        return result_df, selected_features, result_rank, result_cv_score

def run_feature_selection_pipeline(input_file, target_type='ABS'):
    """运行完整的特征选择流程"""
    print(f"Loading data for {target_type} prediction...")
    df = pd.read_excel(input_file)
    
    selector = FeatureSelector()
    
    # 1. 方差选择
    df_var, var_features = selector.variance_selection(df, target_type)
    df_var.to_excel(f'data/processed/{target_type.lower()}_features_var.xlsx', index=False)
    
    # 2. 相关性选择
    df_corr, corr_features = selector.correlation_selection(df_var, target_type)
    df_corr.to_excel(f'data/processed/{target_type.lower()}_features_corr.xlsx', index=False)
    
    # 3. 互信息选择
    df_mi, mi_features, mi_scores = selector.mutual_info_selection(df_corr, target_type)
    df_mi.to_excel(f'data/processed/{target_type.lower()}_features_mi.xlsx', index=False)
    mi_scores.to_excel(f'results/{target_type.lower()}_mi_scores.xlsx', index=False)
    
    # 4. RFECV
    df_rfecv, rfecv_features, ranks, cv_scores = selector.recursive_feature_elimination(
        df_mi, target_type, n_runs=100, model_type='LGB'
    )
    df_rfecv.to_excel(f'data/processed/{target_type.lower()}_features_rfecv.xlsx', index=False)
    ranks.to_excel(f'results/{target_type.lower()}_rfecv_ranks.xlsx')
    cv_scores.to_excel(f'results/{target_type.lower()}_rfecv_scores.xlsx')
    
    print("Feature selection pipeline completed!")

if __name__ == "__main__":
    run_feature_selection_pipeline('data/processed/absorption_features.xlsx', 'ABS')