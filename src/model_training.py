# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:02:17 2023

@author: pc
"""

"""
分子光谱预测模型训练和评估模块
支持模型: KNN, KernelRidge, SVR, RandomForest, XGBoost, LightGBM
"""
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

class ModelTrainer:
    """分子光谱预测模型训练器"""
    
    def __init__(self):
        self.solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
        self.models = {}
        self.best_params = {}
        self.results = {}
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def get_fingerprint_columns(self, df):
        """获取指纹特征列"""
        return [col for col in df.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    
    def create_stratified_folds(self, y, n_splits=10):
        """创建分层折 - 用于回归问题的分层交叉验证"""
        y_max, y_min = y.max(), y.min()
        y_gap = (y_max - y_min) / n_splits
        return [int((val - y_min) // y_gap) for val in y]
    
    def grid_search_knn(self, X, y, n_splits=10):
        """KNN网格搜索"""
        print("Performing KNN grid search...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hash = self.create_stratified_folds(y, n_splits)
        
        n_choices = range(3, 21)
        best_score, best_n = -np.inf, None
        
        for n in tqdm(n_choices, desc="KNN Grid Search"):
            reg = KNeighborsRegressor(n_neighbors=n)
            r2_scores = []
            
            for train_idx, test_idx in skf.split(X, y_hash):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                r2_scores.append(r2_score(y_test, y_pred))
            
            mean_r2 = np.mean(r2_scores)
            if mean_r2 > best_score:
                best_score, best_n = mean_r2, n
        
        self.best_params['KNN'] = {'n_neighbors': best_n}
        print(f"KNN - Best R²: {best_score:.4f}, Best n_neighbors: {best_n}")
        return KNeighborsRegressor(n_neighbors=best_n)
    
    def grid_search_kernel_ridge(self, X, y, n_splits=10):
        """Kernel Ridge网格搜索"""
        print("Performing Kernel Ridge grid search...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hash = self.create_stratified_folds(y, n_splits)
        
        alpha_choices = [10**i for i in range(-5, 2)]
        kernel_choices = ['linear', 'rbf', 'polynomial', 'sigmoid']
        
        best_score, best_alpha, best_kernel = -np.inf, None, None
        
        for alpha in tqdm(alpha_choices, desc="Kernel Ridge Grid Search"):
            for kernel in kernel_choices:
                reg = KernelRidge(alpha=alpha, kernel=kernel)
                r2_scores = []
                
                for train_idx, test_idx in skf.split(X, y_hash):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    r2_scores.append(r2_score(y_test, y_pred))
                
                mean_r2 = np.mean(r2_scores)
                if mean_r2 > best_score:
                    best_score, best_alpha, best_kernel = mean_r2, alpha, kernel
        
        self.best_params['KernelRidge'] = {'alpha': best_alpha, 'kernel': best_kernel}
        print(f"KernelRidge - Best R²: {best_score:.4f}, Best alpha: {best_alpha}, Best kernel: {best_kernel}")
        return KernelRidge(alpha=best_alpha, kernel=best_kernel)
    
    def grid_search_svr(self, X, y, n_splits=10):
        """SVR网格搜索"""
        print("Performing SVR grid search...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hash = self.create_stratified_folds(y, n_splits)
        
        # 扩展的参数搜索空间
        c_choices = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1] + [i*2 for i in range(1, 101)]
        gamma_choices = [2**i for i in range(-5, 3)]
        
        best_score, best_c, best_gamma = -np.inf, None, None
        
        # 限制搜索数量以提高效率
        search_c = [0.125, 1, 10, 100, 200]
        search_gamma = [0.03125, 0.125, 0.5, 2]
        
        for c in tqdm(search_c, desc="SVR Grid Search"):
            for gamma in search_gamma:
                reg = svm.SVR(C=c, gamma=gamma, kernel='rbf')
                r2_scores = []
                
                for train_idx, test_idx in skf.split(X, y_hash):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    r2_scores.append(r2_score(y_test, y_pred))
                
                mean_r2 = np.mean(r2_scores)
                if mean_r2 > best_score:
                    best_score, best_c, best_gamma = mean_r2, c, gamma
        
        self.best_params['SVR'] = {'C': best_c, 'gamma': best_gamma}
        print(f"SVR - Best R²: {best_score:.4f}, Best C: {best_c}, Best gamma: {best_gamma}")
        return svm.SVR(C=best_c, gamma=best_gamma, kernel='rbf')
    
    def grid_search_rf(self, X, y, n_splits=10):
        """Random Forest网格搜索"""
        print("Performing Random Forest grid search...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hash = self.create_stratified_folds(y, n_splits)
        
        n_estimators_choices = range(50, 301, 50)
        max_depth_choices = range(5, 16, 2)
        
        best_score, best_n_est, best_max_depth = -np.inf, None, None
        
        for n_est in tqdm(n_estimators_choices, desc="RF Grid Search"):
            for max_depth in max_depth_choices:
                reg = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, random_state=42)
                r2_scores = []
                
                for train_idx, test_idx in skf.split(X, y_hash):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    r2_scores.append(r2_score(y_test, y_pred))
                
                mean_r2 = np.mean(r2_scores)
                if mean_r2 > best_score:
                    best_score, best_n_est, best_max_depth = mean_r2, n_est, max_depth
        
        self.best_params['RF'] = {'n_estimators': best_n_est, 'max_depth': best_max_depth}
        print(f"Random Forest - Best R²: {best_score:.4f}, Best n_estimators: {best_n_est}, Best max_depth: {best_max_depth}")
        return RandomForestRegressor(n_estimators=best_n_est, max_depth=best_max_depth, random_state=42)
    
    def grid_search_xgboost(self, X, y, n_splits=10):
        """XGBoost网格搜索"""
        print("Performing XGBoost grid search...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hash = self.create_stratified_folds(y, n_splits)
        
        learning_rate_choices = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        max_depth_choices = [3, 5, 7, 9]
        gamma_choices = [0, 0.1, 0.2, 0.3]
        
        best_score, best_lr, best_md, best_gamma = -np.inf, None, None, None
        
        for lr in tqdm(learning_rate_choices, desc="XGBoost Grid Search"):
            for md in max_depth_choices:
                for gamma in gamma_choices:
                    reg = xgb.XGBRegressor(
                        learning_rate=lr, 
                        max_depth=md, 
                        gamma=gamma,
                        random_state=42
                    )
                    r2_scores = []
                    
                    for train_idx, test_idx in skf.split(X, y_hash):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        reg.fit(X_train, y_train)
                        y_pred = reg.predict(X_test)
                        r2_scores.append(r2_score(y_test, y_pred))
                    
                    mean_r2 = np.mean(r2_scores)
                    if mean_r2 > best_score:
                        best_score, best_lr, best_md, best_gamma = mean_r2, lr, md, gamma
        
        self.best_params['XGBoost'] = {
            'learning_rate': best_lr, 
            'max_depth': best_md, 
            'gamma': best_gamma
        }
        print(f"XGBoost - Best R²: {best_score:.4f}, Best learning_rate: {best_lr}, Best max_depth: {best_md}, Best gamma: {best_gamma}")
        return xgb.XGBRegressor(learning_rate=best_lr, max_depth=best_md, gamma=best_gamma, random_state=42)
    
    def grid_search_lightgbm(self, X, y, n_splits=10):
        """LightGBM网格搜索"""
        print("Performing LightGBM grid search...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hash = self.create_stratified_folds(y, n_splits)
        
        learning_rate_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        max_depth_choices = [3, 5, 7, 9]
        num_leaves_choices = [6, 10, 20, 30]
        feature_fraction_choices = [0.6, 0.7, 0.8, 0.9]
        
        best_score, best_lr, best_md, best_nl, best_ff = -np.inf, None, None, None, None
        
        # 限制搜索组合数量
        search_combinations = []
        for lr in [0.2, 0.4, 0.6]:
            for md in [3, 5, 7]:
                for nl in [6, 20, 30]:
                    for ff in [0.7, 0.8, 0.9]:
                        search_combinations.append((lr, md, nl, ff))
        
        for lr, md, nl, ff in tqdm(search_combinations, desc="LightGBM Grid Search"):
            reg = lgb.LGBMRegressor(
                objective='regression',
                learning_rate=lr,
                max_depth=md,
                num_leaves=nl,
                feature_fraction=ff,
                random_state=42
            )
            r2_scores = []
            
            for train_idx, test_idx in skf.split(X, y_hash):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                r2_scores.append(r2_score(y_test, y_pred))
            
            mean_r2 = np.mean(r2_scores)
            if mean_r2 > best_score:
                best_score, best_lr, best_md, best_nl, best_ff = mean_r2, lr, md, nl, ff
        
        self.best_params['LightGBM'] = {
            'learning_rate': best_lr,
            'max_depth': best_md,
            'num_leaves': best_nl,
            'feature_fraction': best_ff
        }
        print(f"LightGBM - Best R²: {best_score:.4f}, Best learning_rate: {best_lr}, Best max_depth: {best_md}, Best num_leaves: {best_nl}, Best feature_fraction: {best_ff}")
        return lgb.LGBMRegressor(
            objective='regression',
            learning_rate=best_lr,
            max_depth=best_md,
            num_leaves=best_nl,
            feature_fraction=best_ff,
            random_state=42
        )
    
    def perform_grid_search_all_models(self, X, y, n_splits=10):
        """对所有模型执行网格搜索"""
        print("=" * 60)
        print("Starting Grid Search for All Models")
        print("=" * 60)
        
        # KNN
        knn_model = self.grid_search_knn(X, y, n_splits)
        
        # Kernel Ridge
        kr_model = self.grid_search_kernel_ridge(X, y, n_splits)
        
        # SVR
        svr_model = self.grid_search_svr(X, y, n_splits)
        
        # Random Forest
        rf_model = self.grid_search_rf(X, y, n_splits)
        
        # XGBoost
        xgb_model = self.grid_search_xgboost(X, y, n_splits)
        
        # LightGBM
        lgb_model = self.grid_search_lightgbm(X, y, n_splits)
        
        self.models = {
            'KNN': knn_model,
            'KernelRidge': kr_model,
            'SVR': svr_model,
            'RF': rf_model,
            'XGBoost': xgb_model,
            'LightGBM': lgb_model
        }
        
        print("\n" + "=" * 60)
        print("Grid Search Completed for All Models")
        print("=" * 60)
        
        return self.models
    
    def cross_validate(self, model, X, y, n_splits, model_name, target_type):
        """执行交叉验证"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_hash = self.create_stratified_folds(y, n_splits)
        
        r2_scores, rmse_scores, mae_scores = [], [], []
        all_predictions = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_hash)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 计算指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
            # 保存预测结果
            fold_df = pd.DataFrame({
                'ID': X_test.index,
                'y_true': y_test.values,
                'y_pred': y_pred
            })
            all_predictions.append(fold_df)
            
            # 保存每个fold的详细结果
            fold_df.to_excel(f'results/{target_type}_{model_name}_fold_{fold+1}.xlsx', index=False)
        
        # 汇总结果
        summary = {
            'R2_mean': np.mean(r2_scores),
            'R2_std': np.std(r2_scores),
            'RMSE_mean': np.mean(rmse_scores),
            'RMSE_std': np.std(rmse_scores),
            'MAE_mean': np.mean(mae_scores),
            'MAE_std': np.std(mae_scores),
            'fold_results': all_predictions,
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores,
            'mae_scores': mae_scores
        }
        
        return summary
    
    def train_and_evaluate_all_models(self, data_file, target_type='ABS', n_splits=10):
        """训练和评估所有模型"""
        print(f"Training and evaluating models for {target_type} prediction...")
        
        # 读取数据
        df = pd.read_excel(data_file)
        fps_cols = self.get_fingerprint_columns(df)
        X = df[self.solvent_cols + fps_cols]
        y = df[target_type]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target variable: {target_type}")
        
        # 网格搜索找到最佳参数
        models = self.perform_grid_search_all_models(X, y, n_splits)
        
        # 交叉验证评估
        self.results = {}
        for name, model in models.items():
            print(f"\nEvaluating {name} with cross-validation...")
            cv_results = self.cross_validate(model, X, y, n_splits, name, target_type)
            self.results[name] = cv_results
            
            # 打印结果
            print(f"{name} - R²: {cv_results['R2_mean']:.4f} ± {cv_results['R2_std']:.4f}")
            print(f"{name} - RMSE: {cv_results['RMSE_mean']:.4f} ± {cv_results['RMSE_std']:.4f}")
            print(f"{name} - MAE: {cv_results['MAE_mean']:.4f} ± {cv_results['MAE_std']:.4f}")
        
        # 保存结果和可视化
        self.save_results(target_type)
        self.plot_results(target_type)
        
        return self.results
    
    def save_results(self, target_type):
        """保存结果到文件"""
        # 保存性能指标
        performance_data = []
        for model_name, result in self.results.items():
            performance_data.append({
                'Model': model_name,
                'R2_mean': result['R2_mean'],
                'R2_std': result['R2_std'],
                'RMSE_mean': result['RMSE_mean'],
                'RMSE_std': result['RMSE_std'],
                'MAE_mean': result['MAE_mean'],
                'MAE_std': result['MAE_std']
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_excel(f'results/{target_type}_model_performance.xlsx', index=False)
        
        # 保存最佳参数
        best_params_df = pd.DataFrame.from_dict(self.best_params, orient='index')
        best_params_df.to_excel(f'results/{target_type}_best_parameters.xlsx')
        
        print(f"\nResults saved to results/{target_type}_* files")
    
    def plot_results(self, target_type):
        """绘制结果可视化"""
        # 性能比较图
        models = list(self.results.keys())
        r2_means = [self.results[model]['R2_mean'] for model in models]
        r2_stds = [self.results[model]['R2_std'] for model in models]
        
        plt.figure(figsize=(12, 8))
        
        # R²比较
        plt.subplot(2, 2, 1)
        y_pos = np.arange(len(models))
        plt.barh(y_pos, r2_means, xerr=r2_stds, align='center', alpha=0.8)
        plt.yticks(y_pos, models)
        plt.xlabel('R² Score')
        plt.title(f'{target_type} Prediction - R² Comparison')
        
        # 预测 vs 真实值散点图 (使用最佳模型)
        best_model = max(self.results.items(), key=lambda x: x[1]['R2_mean'])[0]
        best_results = self.results[best_model]
        
        # 合并所有fold的预测结果
        all_predictions = pd.concat(best_results['fold_results'])
        
        plt.subplot(2, 2, 2)
        plt.scatter(all_predictions['y_true'], all_predictions['y_pred'], alpha=0.6)
        plt.plot([all_predictions['y_true'].min(), all_predictions['y_true'].max()],
                 [all_predictions['y_true'].min(), all_predictions['y_true'].max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{best_model} - True vs Predicted ({target_type})')
        
        # 误差分布
        plt.subplot(2, 2, 3)
        errors = all_predictions['y_pred'] - all_predictions['y_true']
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'{best_model} - Error Distribution')
        
        # 模型比较热力图
        plt.subplot(2, 2, 4)
        metrics = ['R2_mean', 'RMSE_mean', 'MAE_mean']
        comparison_data = []
        for model in models:
            row = [self.results[model][metric] for metric in metrics]
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data, index=models, columns=['R²', 'RMSE', 'MAE'])
        sns.heatmap(comparison_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
        plt.title('Model Performance Comparison')
        
        plt.tight_layout()
        plt.savefig(f'results/{target_type}_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to results/{target_type}_model_comparison.png")

def main():
    """主函数"""
    trainer = ModelTrainer()
    
    # 训练吸收波长预测模型
    print("Starting Absorption Wavelength Prediction...")
    results_abs = trainer.train_and_evaluate_all_models(
        'data/processed/absorption_features_rfecv.xlsx', 
        'ABS', 
        n_splits=10
    )
    
    print("\n" + "=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()