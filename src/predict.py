# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:11:33 2025

@author: pc
"""


"""
分子光谱性质预测模块 - 简化版
直接使用训练好的模型进行预测
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.kernel_ridge import KernelRidge
import joblib

def predict_absorption(pred_data_path, model_path=None):
    """
    预测吸收波长 (ABS)
    
    Parameters:
    - pred_data_path: 预测数据文件路径
    - model_path: 模型文件路径，如果为None则使用默认模型
    
    Returns:
    - predictions: 预测结果数组
    """
    print("预测吸收波长...")
    
    # 加载预测数据
    pred_data = pd.read_excel(pred_data_path)
    
    # 特征列
    solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
    fps_cols = [col for col in pred_data.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    feature_cols = solvent_cols + fps_cols
    
    # 准备特征
    X_pred = pred_data[feature_cols]
    
    # 加载或创建模型
    if model_path:
        model = joblib.load(model_path)
    else:
        # 默认使用KernelRidge模型
        model = KernelRidge(alpha=0.01, kernel='rbf')
        # 如果需要训练模型，请先调用train_absorption_model函数
    
    # 预测
    predictions = model.predict(X_pred)
    
    print(f"预测完成! 样本数量: {len(predictions)}")
    print(f"预测范围: {predictions.min():.1f} - {predictions.max():.1f}")
    
    return predictions

def predict_emission(pred_data_path, model_path=None):
    """
    预测发射波长 (EM)
    """
    print("预测发射波长...")
    
    pred_data = pd.read_excel(pred_data_path)
    solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
    fps_cols = [col for col in pred_data.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    feature_cols = solvent_cols + fps_cols
    
    X_pred = pred_data[feature_cols]
    
    if model_path:
        model = joblib.load(model_path)
    else:
        model = KernelRidge(alpha=0.1, kernel='linear')
    
    predictions = model.predict(X_pred)
    
    print(f"预测完成! 样本数量: {len(predictions)}")
    return predictions

def predict_molar_absorptivity(pred_data_path, model_path=None):
    """
    预测摩尔吸光系数 (LOG)
    """
    print("预测摩尔吸光系数...")
    
    pred_data = pd.read_excel(pred_data_path)
    solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
    fps_cols = [col for col in pred_data.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    feature_cols = solvent_cols + fps_cols
    
    X_pred = pred_data[feature_cols]
    
    if model_path:
        model = joblib.load(model_path)
    else:
        model = xgb.XGBRegressor(learning_rate=0.25, max_depth=6, gamma=0.01)
    
    predictions = model.predict(X_pred)
    
    print(f"预测完成! 样本数量: {len(predictions)}")
    return predictions

def train_absorption_model(train_data_path, save_path=None):
    """
    训练吸收波长模型
    """
    print("训练吸收波长模型...")
    
    train_data = pd.read_excel(train_data_path)
    solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
    fps_cols = [col for col in train_data.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    feature_cols = solvent_cols + fps_cols
    
    X_train = train_data[feature_cols]
    y_train = train_data['ABS']
    
    model = KernelRidge(alpha=0.01, kernel='rbf')
    model.fit(X_train, y_train)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"模型已保存至: {save_path}")
    
    return model

def train_emission_model(train_data_path, save_path=None):
    """
    训练发射波长模型
    """
    print("训练发射波长模型...")
    
    train_data = pd.read_excel(train_data_path)
    solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
    fps_cols = [col for col in train_data.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    feature_cols = solvent_cols + fps_cols
    
    X_train = train_data[feature_cols]
    y_train = train_data['EM']
    
    model = KernelRidge(alpha=0.1, kernel='linear')
    model.fit(X_train, y_train)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"模型已保存至: {save_path}")
    
    return model

def train_molar_absorptivity_model(train_data_path, save_path=None):
    """
    训练摩尔吸光系数模型
    """
    print("训练摩尔吸光系数模型...")
    
    train_data = pd.read_excel(train_data_path)
    solvent_cols = ['Et(30)', 'SP', 'SdP', 'SA', 'SB']
    fps_cols = [col for col in train_data.columns if col.startswith(("MACCS", "Morgan", "RDKit"))]
    feature_cols = solvent_cols + fps_cols
    
    X_train = train_data[feature_cols]
    y_train = train_data['LOG']
    
    model = xgb.XGBRegressor(learning_rate=0.25, max_depth=6, gamma=0.01)
    model.fit(X_train, y_train)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"模型已保存至: {save_path}")
    
    return model

def save_predictions(predictions, pred_data_path, output_path, target_name="ABS"):
    """
    保存预测结果到Excel文件
    """
    pred_data = pd.read_excel(pred_data_path)
    
    # 创建结果DataFrame
    if 'ID' in pred_data.columns:
        result_df = pred_data[['ID', 'SMILES', 'SOLVENT']].copy()
    else:
        result_df = pred_data[['SMILES', 'SOLVENT']].copy()
    
    result_df[f'Predicted_{target_name}'] = predictions
    result_df[f'Predicted_{target_name}'] = result_df[f'Predicted_{target_name}'].round(2)
    
    result_df.to_excel(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")
    
    return result_df

# 使用示例
if __name__ == "__main__":
    # 示例1: 预测吸收波长
    print("=== 吸收波长预测 ===")
    abs_predictions = predict_absorption('dataset/预测/吸收_fps.xlsx')
    save_predictions(abs_predictions, 'dataset/预测/吸收_fps.xlsx', 'results/absorption_predictions.xlsx', 'ABS')
    
    # 示例2: 预测发射波长
    print("\n=== 发射波长预测 ===")
    em_predictions = predict_emission('dataset/预测/发射_fps.xlsx')
    save_predictions(em_predictions, 'dataset/预测/发射_fps.xlsx', 'results/emission_predictions.xlsx', 'EM')
    
    # 示例3: 预测摩尔吸光系数
    print("\n=== 摩尔吸光系数预测 ===")
    log_predictions = predict_molar_absorptivity('dataset/预测/发射_fps.xlsx')
    save_predictions(log_predictions, 'dataset/预测/发射_fps.xlsx', 'results/molar_predictions.xlsx', 'LOG')
    
    print("\n所有预测完成!")