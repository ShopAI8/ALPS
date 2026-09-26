import argparse
import os
import glob
import pandas as pd
import numpy as np
import time
from datetime import datetime
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Optional dependencies for ONNX export
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

try:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False

# ==========================================
# 1. Global configuration
# ==========================================
## "Amazon","BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"
DATASET_LIST = ["Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"]
GENERALIZATION_TARGET_DATASETS = DATASET_LIST.copy()
BASE_DIR = os.environ.get(
    "ALPS_RESULTS_DIR",
    "/your_path/FilterVector/FilterVectorResults"
)
EDA_ROOT_DIR = os.path.join(BASE_DIR, "EDA_Plots_try")

# MODELS_TO_TRY = ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]
# MODELS_TO_TRY = ["XGBoost"]
DEFAULT_ROUTER_MODEL = "XGBoost"
MIN_RECALL_THRESHOLD = 0.90
# Per-dataset min-recall overrides. Fall back to MIN_RECALL_THRESHOLD when unset.
DATASET_MIN_RECALL_THRESHOLDS = {
    "Laion": 0.90,
}
MARGIN_THRESHOLD = 0 
DATASET_MARGIN_THRESHOLDS = {
    "Tiktok": 0,
    "Reviews": 0,
}
USE_SMOTE = False

# Retry once with looser labeling thresholds when pairwise training collapses
# into a single class. This does not affect normally trainable cases.
SINGLE_LABEL_RETRY_MIN_RECALL = 0.9
SINGLE_LABEL_RETRY_MARGIN_THRESHOLD = 0

ROUTE_STRATEGY = {
    "default": "auto"
}

# Training mode configuration
RUN_SINGLE_DATASET_TRAINING = False
RUN_MULTI_DATASET_GENERALIZATION = True # Train one shared model with 100% of the valid samples from every dataset
RUN_CROSS_DATASET_HOLDOUT = False # Train on the other 7 datasets only, then evaluate on the target dataset as a pure holdout
GENERALIZATION_HOLDOUT_TARGET_DATASETS = DATASET_LIST.copy()

# Output paths. Results for the 8-dataset generalization runs are stored under models_8datasets.
SELECT_MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models_8datasets")
SELECT_MODELS_RUN_DIR = "routing_models"
SUMMARY_ROOT_DIR = os.path.join(BASE_DIR, "models_8datasets")
SUMMARY_RUN_DIR = "summary"
LEAVE1_SELECT_MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models_8datasets_leave1")
LEAVE1_SUMMARY_ROOT_DIR = os.path.join(BASE_DIR, "models_8datasets_leave1")

SUPPORTED_ROUTING_CONFIGS = ["FAVOR"]

# Default routing configs to train.
# Supported values: "FAVOR", "ACORN-gamma", "NaviX"
# Example:
AVAILABLE_ROUTING_CONFIGS = ["FAVOR"]

# Whether to train pairwise "two-out-of-three" models by default.
# True:
#   Train only FAVOR vs UNG+, FAVOR vs pre-filter, and UNG+ vs pre-filter
# False:
#   Train both the original 3-way model and the three pairwise models above
DEFAULT_PAIRWISE_ONLY = False

# Whether to train only the original 3-way model by default.
# True:
#   Train only the full candidate set, such as FAVOR / UNG+ / pre-filter
# False:
#   Pairwise training is controlled by DEFAULT_PAIRWISE_ONLY and CLI flags
DEFAULT_FULL_MODEL_ONLY = True

# ==========================================
# 2. Core utilities and features
# ==========================================
def create_classifier(model_type="RandomForest", **kwargs):
    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        params = {'n_estimators': 150, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params.update(kwargs)
        return RandomForestClassifier(**params)
        
    elif model_type == "LightGBM":
        import lightgbm as lgb
        params = {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 200, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced', 'verbose': -1}
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)
        
    elif model_type == "XGBoost":
        import xgboost as xgb
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 1,
            'colsample_bytree': 1,
            'min_child_weight': 1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }

        params.update(kwargs)
        return xgb.XGBClassifier(**params)
        
    elif model_type == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        params = {'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}
        params.update(kwargs)
        return DecisionTreeClassifier(**params)
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def infer_algorithm_list(df, time_prefix='L1_Time_ms'):
    algo_list = []
    prefix = f"{time_prefix}_"
    for col in df.columns:
        if col.startswith(prefix):
            algo = col[len(prefix):]
            recall_col = f"Recall_{algo}"
            if recall_col in df.columns:
                algo_list.append(algo)
    return sorted(algo_list)

def sanitize_name(name):
    return (
        str(name)
        .replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )

def build_candidate_set_name(algo_list):
    return "__vs__".join(sanitize_name(algo) for algo in algo_list)

def build_dataset_output_dir(dataset_name, config_name):
    return os.path.join(
        SELECT_MODELS_ROOT_DIR,
        dataset_name,
        SELECT_MODELS_RUN_DIR,
        config_name
    )

def build_summary_output_dir(config_name):
    return os.path.join(SUMMARY_ROOT_DIR, SUMMARY_RUN_DIR, config_name)

def build_leave1_summary_output_dir(config_name):
    return os.path.join(LEAVE1_SUMMARY_ROOT_DIR, SUMMARY_RUN_DIR, config_name)

def get_min_recall_threshold(dataset_name):
    return DATASET_MIN_RECALL_THRESHOLDS.get(dataset_name, MIN_RECALL_THRESHOLD)

def get_margin_threshold(dataset_name):
    return DATASET_MARGIN_THRESHOLDS.get(dataset_name, MARGIN_THRESHOLD)

def label_best_algorithm_by_time(df, algo_list, time_prefix='L1_Time_ms', min_recall=MIN_RECALL_THRESHOLD, threshold=0.15):
    best_algos = []
    fuzzy_count = 0
    for idx, row in df.iterrows():
        candidates = []
        for algo in algo_list:
            recall_col = f'Recall_{algo}'
            time_col = f'{time_prefix}_{algo}'
            if recall_col in row and time_col in row and pd.notna(row[recall_col]) and pd.notna(row[time_col]):
                candidates.append({'algo': algo, 'recall': row[recall_col], 'time': row[time_col]})
                
        if not candidates:
            best_algos.append('Unknown')
            continue
            
        qualified = [c for c in candidates if c['recall'] >= min_recall]
        
        if not qualified:
            best = max(candidates, key=lambda x: x['recall'])
            best_algos.append(best['algo'])
        else:
            qualified.sort(key=lambda x: x['time'])
            best = qualified[0]
            
            if len(qualified) > 1:
                second_best = qualified[1]
                time_diff_percent = (second_best['time'] - best['time']) / (best['time'] + 1e-9)
                if time_diff_percent < threshold:
                    best_algos.append('Unknown')
                    fuzzy_count += 1
                    continue
                    
            best_algos.append(best['algo'])
            
    return pd.Series(best_algos, index=df.index)

def generate_features(df):
    # Build single-layer routing features.
    X = pd.DataFrame(index=df.index)
    X['GlobalPpass'] = df['GlobalPpass']
    X['NumDescendants'] = df['NumDescendants']
    X['QuerySize'] = df['QuerySize']
    # X['CandSize'] = df['CandSize']
    
    # Normalize invalid values (NaN, inf).
    X.replace([np.inf, -np.inf], np.nan, inplace=True); X.fillna(0, inplace=True)
    
    return X

def prepare_routing_training_data(df, dataset_name, algo_list, route_scope_name):
    if len(algo_list) < 2:
        print(f"❌ 候选算法数量不足 2 个: {algo_list}，跳过 {route_scope_name}。")
        return None

    missing_cols = []
    for algo in algo_list:
        time_col = f"L1_Time_ms_{algo}"
        recall_col = f"Recall_{algo}"
        if time_col not in df.columns:
            missing_cols.append(time_col)
        if recall_col not in df.columns:
            missing_cols.append(recall_col)

    if missing_cols:
        print(f"❌ 缺少候选算法所需列，跳过 {route_scope_name}: {missing_cols}")
        return None

    min_recall_threshold = get_min_recall_threshold(dataset_name)
    margin_threshold = get_margin_threshold(dataset_name)
    print(f"🎯 {route_scope_name} 使用 min_recall={min_recall_threshold}, margin={margin_threshold}")

    work_df = df.copy()
    work_df['Global_Best'] = label_best_algorithm_by_time(
        work_df, algo_list, time_prefix='L1_Time_ms', min_recall=min_recall_threshold, threshold=margin_threshold
    )
    work_df['Target'] = work_df['Global_Best']
    unknown_count = int((work_df['Global_Best'] == 'Unknown').sum())
    total_count = int(len(work_df))
    unknown_rate = (unknown_count / total_count) if total_count else 0.0

    X = generate_features(work_df)
    valid_mask = (work_df['Global_Best'] != 'Unknown')
    valid_indices = work_df[valid_mask].index

    if len(valid_indices) < 2:
        print(f"❌ {route_scope_name} 有效样本不足，跳过。")
        return None

    y_valid = work_df.loc[valid_indices, 'Target']
    if y_valid.nunique() < 2:
        print(
            f"⚠️ {route_scope_name} 当前阈值下仅存在单一标签 {sorted(y_valid.unique())}，"
            f"尝试使用更宽松阈值重打标: min_recall={SINGLE_LABEL_RETRY_MIN_RECALL}, "
            f"margin={SINGLE_LABEL_RETRY_MARGIN_THRESHOLD}"
        )
        retry_df = df.copy()
        retry_df['Global_Best'] = label_best_algorithm_by_time(
            retry_df,
            algo_list,
            time_prefix='L1_Time_ms',
            min_recall=SINGLE_LABEL_RETRY_MIN_RECALL,
            threshold=SINGLE_LABEL_RETRY_MARGIN_THRESHOLD
        )
        retry_df['Target'] = retry_df['Global_Best']
        retry_valid_mask = (retry_df['Global_Best'] != 'Unknown')
        retry_valid_indices = retry_df[retry_valid_mask].index
        retry_y_valid = retry_df.loc[retry_valid_indices, 'Target']

        if len(retry_valid_indices) < 2 or retry_y_valid.nunique() < 2:
            print(f"❌ {route_scope_name} 在放宽阈值后仍仅存在单一标签 {sorted(retry_y_valid.unique())}，跳过。")
            return None

        work_df = retry_df
        X = generate_features(work_df)
        valid_indices = retry_valid_indices
        y_valid = retry_y_valid
        unknown_count = int((work_df['Global_Best'] == 'Unknown').sum())
        total_count = int(len(work_df))
        unknown_rate = (unknown_count / total_count) if total_count else 0.0
        print(f"✅ {route_scope_name} 放宽阈值后恢复为可训练数据。")

    return {
        "work_df": work_df,
        "X": X,
        "valid_indices": valid_indices,
        "y_valid": y_valid,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "total_count": total_count
    }

def fit_routing_model(X_train, y_train, target_map, model_type, use_smote=False):
    """Fit a router without evaluating it on the training samples."""
    y_train_mapped = y_train.map(target_map)
    train_mask = y_train_mapped.notna()
    X_train_clean = X_train[train_mask].copy()
    y_train_clean = y_train_mapped[train_mask].astype(int)

    if X_train_clean.empty:
        raise ValueError("训练集没有可用的已标注样本。")
    
    train_unique = sorted(y_train_clean.unique())
    remapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(train_unique)}
    y_train_cont = y_train_clean.map(remapping).astype(int)
    
    internal_to_original = {new_lbl: old_lbl for old_lbl, new_lbl in remapping.items()}
    real_classes = [internal_to_original[i] for i in range(len(internal_to_original))]

    if use_smote:
        try:
            min_samples = y_train_cont.value_counts().min()
            safe_k = min(5, min_samples - 1) if min_samples > 1 else 1
            if min_samples > 1:
                smote = SMOTE(random_state=42, k_neighbors=safe_k)
                X_train_clean, y_train_cont = smote.fit_resample(X_train_clean, y_train_cont)
        except Exception:
            pass

    X_train_np = X_train_clean.values
    y_train_np = y_train_cont.values

    classifier = create_classifier(model_type=model_type)

    t_train_start = time.perf_counter()
    classifier.fit(X_train_np, y_train_np)
    train_time_ms = (time.perf_counter() - t_train_start) * 1000.0

    importances_dict = {}
    if hasattr(classifier, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': classifier.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        importance_str = importance_df.to_string(formatters={'Importance': '{:.4f}'.format})
        importances_dict = dict(zip(X_train.columns, classifier.feature_importances_))
    else:
        importance_str = "当前模型不支持提取 Feature Importances。"

    return {
        "train_time_ms": train_time_ms,
        "importance_str": importance_str,
        "importances_dict": importances_dict,
        "classifier": classifier,
        "real_classes": real_classes,
        "train_size": len(X_train_np)
    }

def train_and_evaluate_model(X_train, y_train, X_test, y_test, target_map, model_type, use_smote=False):
    fit_result = fit_routing_model(
        X_train,
        y_train,
        target_map,
        model_type,
        use_smote=use_smote
    )
    evaluation = evaluate_trained_model(
        X_test,
        y_test,
        target_map,
        fit_result["classifier"],
        fit_result["real_classes"]
    )
    return {**fit_result, **evaluation}

def evaluate_trained_model(X_test, y_test, target_map, classifier, real_classes):
    """Evaluate one already-fitted shared router without retraining it."""
    y_test_mapped = y_test.map(target_map)
    test_mask = y_test_mapped.notna()
    X_test_clean = X_test[test_mask].copy()
    y_test_clean = y_test_mapped[test_mask].astype(int)

    if X_test_clean.empty:
        raise ValueError("测试集没有可评估的有效样本。")

    t_pred_start = time.perf_counter()
    y_pred_np = classifier.predict(X_test_clean.values)
    pred_latency_us = ((time.perf_counter() - t_pred_start) * 1e6) / len(X_test_clean)

    y_pred_abs = np.array([real_classes[int(idx)] for idx in y_pred_np])
    y_test_abs = y_test_clean.values
    acc = accuracy_score(y_test_abs, y_pred_abs)

    inv_map = {v: k for k, v in target_map.items()}
    present_labels = sorted(np.unique(np.concatenate((y_test_abs, y_pred_abs))))
    target_names = [inv_map[label] for label in present_labels]
    cls_report = classification_report(
        y_test_abs,
        y_pred_abs,
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    )
    cm = confusion_matrix(y_test_abs, y_pred_abs, labels=present_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{name}" for name in target_names],
        columns=[f"Pred_{name}" for name in target_names]
    )

    return {
        "acc": acc,
        "pred_latency_us": pred_latency_us,
        "cls_report": cls_report,
        "cm_df": cm_df,
        "test_size": len(X_test_clean)
    }

def run_layer_ablation(X_train, y_train, X_test, y_test, target_map, best_model_type):
    """Run ablations specifically for evidence-collection analysis."""
    results = {}
    # 1. Baseline
    res_base = train_and_evaluate_model(X_train, y_train, X_test, y_test, target_map, best_model_type)
    results["Baseline (All)"] = res_base["acc"]
    
    # 2. Remove each feature group one at a time.
    for col in X_train.columns:
        ablated_cols = [c for c in X_train.columns if c != col]
        if not ablated_cols: continue
        res_abl = train_and_evaluate_model(X_train[ablated_cols], y_train, X_test[ablated_cols], y_test, target_map, best_model_type)
        results[f"Minus {col}"] = res_abl["acc"]
        
    # 3. Stress-test the core signal with GlobalPpass only.
    if 'GlobalPpass' in X_train.columns and len(X_train.columns) > 1:
        res_only_ppass = train_and_evaluate_model(X_train[['GlobalPpass']], y_train, X_test[['GlobalPpass']], y_test, target_map, best_model_type)
        results["Only GlobalPpass"] = res_only_ppass["acc"]
        
    return results

def run_arena_for_layer(X_train, y_train, X_test, y_test, target_map, layer_name, use_smote=False):
    smote_status = "启用" if use_smote else "未启用"
    print(f"\n[{layer_name} - 模型竞技场启动 | SMOTE: {smote_status}]")
    results_cache = {}
    metrics_list = []
    
    for model_type in MODELS_TO_TRY:
        try:
            res = train_and_evaluate_model(X_train, y_train, X_test, y_test, target_map, model_type, use_smote=use_smote)
            results_cache[model_type] = res
            metrics_list.append({
                "Model": model_type,
                "Accuracy": res['acc'],
                "Train_Time_ms": res['train_time_ms'],
                "Pred_Latency_us": res['pred_latency_us'],
                "Test_Size": res['test_size']
            })
            print(f"  > {model_type:<15} | 准确率: {res['acc']:.4%} | 训练耗时: {res['train_time_ms']:.2f} ms")
        except ImportError:
             print(f"  > ⚠️ 缺少 {model_type} 引擎依赖库。")
        except Exception as e:
            print(f"  > ⚠️ {model_type} 训练失败: {e}")
            
    return results_cache, metrics_list

def save_onnx_model(classifier, model_type, num_features, output_dir, filename, real_classes=None):
    onnx_filename = os.path.join(output_dir, filename)
    try:
        if model_type in ["RandomForest", "DecisionTree"] and SKL2ONNX_AVAILABLE:
            from skl2onnx import convert_sklearn
            initial_type = [('float_input', SklFloatTensorType([None, num_features]))]
            onnx_model = convert_sklearn(classifier, initial_types=initial_type, target_opset=15)
        elif model_type in ["LightGBM", "XGBoost"] and ONNXMLTOOLS_AVAILABLE:
            from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
            initial_type = [('float_input', OnnxFloatTensorType([None, num_features]))]
            if model_type == "LightGBM":
                import onnxmltools
                onnx_model = onnxmltools.convert_lightgbm(classifier, initial_types=initial_type, target_opset=15)
            else:
                import onnxmltools
                onnx_model = onnxmltools.convert_xgboost(classifier, initial_types=initial_type, target_opset=15)
        else:
            print(f"⚠️ 无法导出 {model_type} 的 ONNX 模型。")
            return
            
        with open(onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        if real_classes is not None:
            import onnx
            model = onnx.load(onnx_filename)
            patched = False
            for node in model.graph.node:
                for attr in node.attribute:
                    if attr.name == 'classlabels_int64s':
                        del attr.ints[:]
                        attr.ints.extend(real_classes)
                        patched = True
            if patched:
                onnx.save(model, onnx_filename)
                print(f"🔧 [ONNX Hack] {filename} 已成功硬编码底层输出节点。")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型时发生错误: {e}")

# ELS-related inputs and branching logic have been removed.
def calculate_system_accuracy(X_test, y_global_best_test, clf, real_classes, target_map):
    preds_raw = clf.predict(X_test.values)
    inv_target_map = {v: k for k, v in target_map.items()}
    preds_mapped = [inv_target_map[real_classes[int(idx)]] for idx in preds_raw]
            
    valid_mask = y_global_best_test != 'Unknown'
    y_true_valid = y_global_best_test[valid_mask]
    final_preds_valid = [preds_mapped[j] for j, valid in enumerate(valid_mask) if valid]
    
    if len(y_true_valid) == 0:
        return 0.0
    return accuracy_score(y_true_valid, final_preds_valid)

def generate_comparison_table(metrics_list, layer_name, feature_count):
    metrics_list.sort(key=lambda x: x["Accuracy"], reverse=True)
    lines = [f"[{layer_name} - 性能对比 (特征数: {feature_count})]"]
    lines.append(f"  {'算法模型(Model)':<16} | {'准确率(Accuracy)':<16} | {'训练耗时(Train ms)':<18} | {'单次推理(Pred μs)':<18}")
    lines.append("  " + "-" * 76)
    for m in metrics_list:
        lines.append(f"  {m['Model']:<16} | {m['Accuracy']:<18.4%} | {m['Train_Time_ms']:<18.2f} | {m['Pred_Latency_us']:<18.2f}")
    return "\n".join(lines) + "\n"

def build_model_aggregate_summary(df_metrics):
    metric_cols = ["Accuracy", "Train_Time_ms", "Pred_Latency_us"]
    if df_metrics.empty or "Model" not in df_metrics.columns:
        return pd.DataFrame(columns=["Mode", "Model"] + [f"{col}_{stat}" for col in metric_cols for stat in ["Sum", "Mean"]])

    base_models = [model for model in MODELS_TO_TRY if model in set(df_metrics["Model"].dropna())]
    if not base_models:
        return pd.DataFrame(columns=["Mode", "Model"] + [f"{col}_{stat}" for col in metric_cols for stat in ["Sum", "Mean"]])

    df_base = df_metrics[df_metrics["Model"].isin(base_models)].copy()
    if df_base.empty:
        return pd.DataFrame(columns=["Mode", "Model"] + [f"{col}_{stat}" for col in metric_cols for stat in ["Sum", "Mean"]])

    for col in metric_cols:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

    if "Mode" not in df_base.columns:
        df_base["Mode"] = "Default"

    agg_df = (
        df_base.groupby(["Mode", "Model"], as_index=False)[metric_cols]
        .agg(["sum", "mean"])
        .reset_index()
    )
    agg_df.columns = [
        col[0] if col[0] in {"Mode", "Model"} else f"{col[0]}_{col[1].capitalize()}"
        for col in agg_df.columns.to_flat_index()
    ]

    agg_df["Model"] = pd.Categorical(agg_df["Model"], categories=base_models, ordered=True)
    agg_df = agg_df.sort_values(["Mode", "Model"]).reset_index(drop=True)
    agg_df["Model"] = agg_df["Model"].astype(str)
    return agg_df

def write_summary_outputs(global_metrics, global_ablation, global_importances, config_summary_dir):
    print(f"📁 Summary 输出目录: {config_summary_dir}")
    os.makedirs(config_summary_dir, exist_ok=True)

    df_all = pd.DataFrame(global_metrics)
    cols_order = [
        "Mode", "Config", "Dataset", "Train_Datasets", "Layer",
        "Candidate_Set", "Model", "Model_Selection", "Accuracy",
        "Train_Time_ms", "Pred_Latency_us", "Train_Size", "Test_Size",
        "Evaluation_Size", "Evaluation_Scope", "Evaluation_CSV",
        "Unknown_Count", "Unknown_Rate", "Total_Samples"
    ]
    df_all = df_all[[c for c in cols_order if c in df_all.columns]]
    csv1_path = os.path.join(config_summary_dir, "fast_all_datasets_metrics.csv")
    df_all.to_csv(csv1_path, index=False)
    print(f"\n✅ [全局报表 1] 各数据集单层算法明细已保存至: {csv1_path}")

    df_model_agg = build_model_aggregate_summary(df_all)
    csv_model_agg_path = os.path.join(config_summary_dir, "fast_model_aggregate_metrics.csv")
    df_model_agg.to_csv(csv_model_agg_path, index=False)
    print(f"✅ [全局报表 1.1] 各算法 Accuracy/Train_Time_ms/Pred_Latency_us 的总和与平均值已保存至: {csv_model_agg_path}")

    if global_ablation:
        df_abl = pd.DataFrame(global_ablation)
        pivot_abl = df_abl.pivot_table(index=["Config", "Dataset", "Layer"], columns="Config", values="Accuracy").reset_index()
        csv_abl_path = os.path.join(config_summary_dir, "fast_ablation_summary.csv")
        pivot_abl.to_csv(csv_abl_path, index=False)
        print(f"✅ [全局报表 2] 核心特征消融实验汇总已保存至: {csv_abl_path}")

    if global_importances:
        df_imp = pd.DataFrame(global_importances)
        meta_cols = ["Config", "Dataset", "Layer", "Model"]
        feat_cols = [c for c in df_imp.columns if c not in meta_cols]
        df_imp = df_imp[meta_cols + feat_cols]
        csv_imp_path = os.path.join(config_summary_dir, "fast_feature_importances.csv")
        df_imp.to_csv(csv_imp_path, index=False)
        print(f"✅ [全局报表 3] 全局特征重要性得分表已保存至: {csv_imp_path}\n")

def train_routing_model_for_candidates(df, dataset_name, config_name, algo_list, output_dir, report_path, route_scope_name):
    prepared = prepare_routing_training_data(df, dataset_name, algo_list, route_scope_name)
    if prepared is None:
        return [], [], []

    work_df = prepared["work_df"]
    X = prepared["X"]
    valid_indices = prepared["valid_indices"]
    y_valid = prepared["y_valid"]
    unknown_count = prepared["unknown_count"]
    unknown_rate = prepared["unknown_rate"]
    total_count = prepared["total_count"]

    target_map = {algo: idx for idx, algo in enumerate(algo_list)}

    stratify_labels = y_valid if y_valid.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        valid_indices,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels
    )

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = work_df.loc[train_idx, 'Target'], work_df.loc[test_idx, 'Target']
    y_global_best_test = work_df.loc[test_idx, 'Global_Best']

    cache, metrics = run_arena_for_layer(
        X_train, y_train, X_test, y_test, target_map, route_scope_name, use_smote=USE_SMOTE
    )
    if not metrics:
        print(f"❌ {route_scope_name} 没有成功训练出的模型，跳过。")
        return [], [], []

    current_strategy = ROUTE_STRATEGY
    if isinstance(ROUTE_STRATEGY, dict):
        current_strategy = ROUTE_STRATEGY.get(dataset_name, ROUTE_STRATEGY.get("default", "auto"))

    if isinstance(current_strategy, str) and current_strategy != "auto":
        best_model = current_strategy
        strategy_desc = f"Manual ({best_model})"
    else:
        best_model = max(metrics, key=lambda x: x['Accuracy'])['Model']
        strategy_desc = "Auto (自动取优)"

    print(f"\n[策略选定] {route_scope_name} 胜出模型: {best_model}")
    res = cache[best_model]

    dataset_ablation = []
    dataset_importances = []
    candidate_set_name = " vs ".join(algo_list)
    if res['importances_dict']:
        imp = res['importances_dict'].copy()
        imp.update({
            "Dataset": dataset_name,
            "Config": config_name,
            "Layer": route_scope_name,
            "Model": best_model,
            "Candidate_Set": candidate_set_name
        })
        dataset_importances.append(imp)

    system_acc = calculate_system_accuracy(X_test, y_global_best_test, res['classifier'], res['real_classes'], target_map)

    os.makedirs(output_dir, exist_ok=True)
    save_onnx_model(res['classifier'], best_model, X_train.shape[1], output_dir, "router.onnx", res['real_classes'])
    save_class_labels(output_dir, algo_list)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        f.write("┃               Fast Single-Layer Routing Assessment Report          ┃\n")
        f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
        f.write(f"┃ 数据集  : {dataset_name:<15} 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} ┃\n")
        f.write(f"┃ 配置    : {config_name:<52} ┃\n")
        f.write(f"┃ 候选集  : {candidate_set_name:<52} ┃\n")
        f.write(f"┃ 选用策略: {strategy_desc:<52} ┃\n")
        f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")

        f.write("【零 | 模型竞技与特征对比 (Model Arena Comparison)】\n")
        f.write("-" * 80 + "\n")
        f.write(generate_comparison_table(metrics, route_scope_name, X_train.shape[1]) + "\n")

        f.write("【壹 | 端到端表现评估 (End-to-End Evaluation)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  ▶ 选定引擎 : {best_model}\n")
        f.write(f"  ▶ 候选算法 : {', '.join(algo_list)}\n\n")
        f.write(f"  ▶ Unknown 样本数 : {unknown_count} / {total_count} ({unknown_rate:.4%})\n\n")
        f.write(f"  🚀 模拟流水线分发最终准确率 (System End-to-End Accuracy): {system_acc:.4%}\n\n")

        f.write("【叁 | 胜出模型透视 (Selected Model Deep Dive)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"[胜出者: {best_model}]\n")
        f.write("  ▶ 详细分类报告:\n  " + res['cls_report'].replace('\n', '\n  ') + "\n")
        f.write("  ▶ 混淆矩阵:\n  " + res['cm_df'].to_string().replace('\n', '\n  ') + "\n\n")
        f.write("  ▶ 特征重要性:\n  " + res['importance_str'].replace('\n', '\n  ') + "\n\n")

    print(f"\n✅ {route_scope_name} 评估完成！详尽log已同步至: {report_path}")

    dataset_metrics = []
    for m in metrics:
        m_copy = m.copy()
        m_copy.update({
            "Dataset": dataset_name,
            "Config": config_name,
            "Layer": route_scope_name,
            "Candidate_Set": candidate_set_name,
            "Unknown_Count": unknown_count,
            "Unknown_Rate": unknown_rate,
            "Total_Samples": total_count
        })
        dataset_metrics.append(m_copy)

    dataset_metrics.append({
        "Dataset": dataset_name,
        "Config": config_name,
        "Layer": f"{route_scope_name}_System_End_to_End",
        "Model": f"Single({best_model})",
        "Accuracy": system_acc,
        "Train_Time_ms": np.nan,
        "Pred_Latency_us": np.nan,
        "Test_Size": len(y_global_best_test[y_global_best_test != 'Unknown']),
        "Candidate_Set": candidate_set_name,
        "Unknown_Count": unknown_count,
        "Unknown_Rate": unknown_rate,
        "Total_Samples": total_count
    })

    return dataset_metrics, dataset_ablation, dataset_importances

# ==========================================
# 3. Single-dataset processing pipeline
# ==========================================
def save_class_labels(output_dir, algo_list):
    labels_path = os.path.join(output_dir, "class_labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for idx, algo in enumerate(algo_list):
            f.write(f"{idx},{algo}\n")

def process_single_dataset(
    dataset_name,
    config_name,
    train_full_model=True,
    train_pairwise_models=True
):
    print(f"\n{'='*70}")
    print(f"🚀 开始处理单层路由数据集: {dataset_name} | 配置: {config_name}")
    print(f"{'='*70}")
    
    csv_path = os.path.join(EDA_ROOT_DIR, config_name, dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = build_dataset_output_dir(dataset_name, config_name)
    print(f"📁 单模型输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}，跳过。")
        return [], [], []
        
    df = pd.read_csv(csv_path)
    all_metrics = []
    all_ablations = []
    all_importances = []

    algo_list = infer_algorithm_list(df)
    if train_full_model:
        full_report_path = os.path.join(output_dir, f"SmartRoute_Single_Report_{dataset_name}.txt")
        full_metrics, full_ablation, full_importances = train_routing_model_for_candidates(
            df=df,
            dataset_name=dataset_name,
            config_name=config_name,
            algo_list=algo_list,
            output_dir=output_dir,
            report_path=full_report_path,
            route_scope_name="Single Layer (单层路由)"
        )
        all_metrics.extend(full_metrics)
        all_ablations.extend(full_ablation)
        all_importances.extend(full_importances)

    if train_pairwise_models:
        for pair_algos in combinations(algo_list, 2):
            pair_list = list(pair_algos)
            pair_name = build_candidate_set_name(pair_list)
            pair_output_dir = os.path.join(output_dir, "pairwise_models", pair_name)
            pair_report_path = os.path.join(pair_output_dir, f"SmartRoute_Pair_Report_{dataset_name}_{pair_name}.txt")
            pair_scope_name = f"Pairwise ({pair_list[0]} vs {pair_list[1]})"
            print(f"\n--- 开始训练二选一模型: {pair_list[0]} vs {pair_list[1]} ---")
            pair_metrics, pair_ablation, pair_importances = train_routing_model_for_candidates(
                df=df,
                dataset_name=dataset_name,
                config_name=config_name,
                algo_list=pair_list,
                output_dir=pair_output_dir,
                report_path=pair_report_path,
                route_scope_name=pair_scope_name
            )
            all_metrics.extend(pair_metrics)
            all_ablations.extend(pair_ablation)
            all_importances.extend(pair_importances)

    return all_metrics, all_ablations, all_importances

def load_dataset_dataframe(dataset_name, config_name):
    csv_path = os.path.join(EDA_ROOT_DIR, config_name, dataset_name, f"{dataset_name}_aligned_results.csv")
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        return None
    return pd.read_csv(csv_path)

def resolve_external_evaluation_csv(dataset_name, config_name, evaluation_root=None, evaluation_files=None):
    """Resolve one dataset's independent routing-evaluation CSV."""
    evaluation_files = evaluation_files or {}
    if dataset_name in evaluation_files:
        return os.path.abspath(evaluation_files[dataset_name])

    if not evaluation_root:
        return None

    root = os.path.abspath(evaluation_root)
    candidates = [
        os.path.join(root, config_name, dataset_name, f"{dataset_name}_routing_eval.csv"),
        os.path.join(root, config_name, dataset_name, f"{dataset_name}_aligned_results.csv"),
        os.path.join(root, dataset_name, f"{dataset_name}_routing_eval.csv"),
        os.path.join(root, dataset_name, f"{dataset_name}_aligned_results.csv"),
        os.path.join(root, f"{dataset_name}_routing_eval.csv"),
        os.path.join(root, f"{dataset_name}_aligned_results.csv"),
        os.path.join(root, f"{dataset_name}.csv"),
    ]
    return next((path for path in candidates if os.path.isfile(path)), None)

def load_external_evaluation_dataframe(
    dataset_name,
    config_name,
    evaluation_root=None,
    evaluation_files=None
):
    csv_path = resolve_external_evaluation_csv(
        dataset_name,
        config_name,
        evaluation_root=evaluation_root,
        evaluation_files=evaluation_files
    )
    if not csv_path or not os.path.isfile(csv_path):
        print(f"❌ 找不到 {dataset_name} 的独立路由评测 CSV: {csv_path or '未提供'}")
        return None, csv_path
    return pd.read_csv(csv_path), csv_path

def prepare_external_routing_evaluation_data(df, dataset_name, algo_list, label_column):
    """Prepare independent queries whose correct route is supplied by the caller."""
    required_features = ["GlobalPpass", "NumDescendants", "QuerySize"]
    missing_features = [column for column in required_features if column not in df.columns]
    if missing_features:
        raise ValueError(f"{dataset_name} 评测 CSV 缺少特征列: {missing_features}")
    if label_column not in df.columns:
        raise ValueError(
            f"{dataset_name} 评测 CSV 缺少正确路由列 '{label_column}'；"
            f"该列应填写算法名: {algo_list}"
        )

    index_to_algo = {idx: algo for idx, algo in enumerate(algo_list)}

    def normalize_label(value):
        if pd.isna(value):
            return None
        if value in algo_list:
            return value
        text_value = str(value).strip()
        if text_value in algo_list or text_value == "Unknown":
            return text_value
        try:
            numeric_value = float(text_value)
            integer_value = int(numeric_value)
            if numeric_value == integer_value and integer_value in index_to_algo:
                return index_to_algo[integer_value]
        except ValueError:
            pass
        return text_value

    normalized_labels = df[label_column].map(normalize_label)
    valid_mask = normalized_labels.isin(algo_list)
    invalid_labels = sorted({
        str(label) for label in normalized_labels[~valid_mask].dropna().unique()
        if str(label) != "Unknown"
    })
    if invalid_labels:
        print(f"⚠️ {dataset_name} 评测集忽略不在候选集中的标签: {invalid_labels}")

    X_eval = generate_features(df).loc[valid_mask].copy()
    y_eval = normalized_labels.loc[valid_mask].copy()
    if X_eval.empty:
        raise ValueError(f"{dataset_name} 评测 CSV 没有可评估的有效路由标签。")

    return {
        "X": X_eval,
        "y": y_eval,
        "valid_count": len(X_eval),
        "ignored_count": int((~valid_mask).sum()),
        "total_count": len(df)
    }

def select_common_algorithms(train_dataset_names, holdout_dataset_name, config_name):
    algo_sets = []
    ordered_reference = None

    for dataset_name in train_dataset_names + [holdout_dataset_name]:
        df = load_dataset_dataframe(dataset_name, config_name)
        if df is None:
            return None
        current_algos = infer_algorithm_list(df)
        if ordered_reference is None and dataset_name == holdout_dataset_name:
            ordered_reference = current_algos
        algo_sets.append(set(current_algos))

    common_algos = set.intersection(*algo_sets) if algo_sets else set()
    if not common_algos:
        return []

    if ordered_reference is None:
        ordered_reference = sorted(common_algos)
    return [algo for algo in ordered_reference if algo in common_algos]

def select_common_algorithms_for_datasets(dataset_names, config_name):
    if not dataset_names:
        return []
    return select_common_algorithms(dataset_names[:-1], dataset_names[-1], config_name)

def train_cross_dataset_holdout_for_candidates(
    target_dataset_name,
    aux_dataset_names,
    config_name,
    algo_list,
    output_dir,
    report_path,
    route_scope_name
):
    if len(aux_dataset_names) == 0:
        print(f"❌ {route_scope_name} 没有可用的训练数据集。")
        return [], [], []

    target_map = {algo: idx for idx, algo in enumerate(algo_list)}
    train_feature_frames = []
    train_target_frames = []
    train_dataset_stats = []

    for source_dataset in aux_dataset_names:
        df_source = load_dataset_dataframe(source_dataset, config_name)
        if df_source is None:
            continue

        prepared_source = prepare_routing_training_data(
            df_source,
            source_dataset,
            algo_list,
            f"{route_scope_name} | Train Source: {source_dataset}"
        )
        if prepared_source is None:
            continue

        source_indices = prepared_source["valid_indices"]
        train_feature_frames.append(prepared_source["X"].loc[source_indices].copy())
        train_target_frames.append(prepared_source["work_df"].loc[source_indices, "Target"].copy())
        train_dataset_stats.append({
            "Dataset": source_dataset,
            "Valid_Samples": len(source_indices),
            "Unknown_Count": prepared_source["unknown_count"],
            "Unknown_Rate": prepared_source["unknown_rate"],
            "Total_Samples": prepared_source["total_count"]
        })

    if not train_feature_frames:
        print(f"❌ {route_scope_name} 没有可用的跨数据集训练样本。")
        return [], [], []

    df_target = load_dataset_dataframe(target_dataset_name, config_name)
    if df_target is None:
        return [], [], []

    prepared_target = prepare_routing_training_data(
        df_target,
        target_dataset_name,
        algo_list,
        f"{route_scope_name} | Test Target: {target_dataset_name}"
    )
    if prepared_target is None:
        return [], [], []

    X_train = pd.concat(train_feature_frames, axis=0, ignore_index=True)
    y_train = pd.concat(train_target_frames, axis=0, ignore_index=True)

    target_test_indices = prepared_target["valid_indices"]
    X_test = prepared_target["X"].loc[target_test_indices].copy()
    y_test = prepared_target["work_df"].loc[target_test_indices, "Target"].copy()
    y_global_best_test = prepared_target["work_df"].loc[target_test_indices, "Global_Best"]

    if y_train.nunique() < 2:
        print(f"❌ {route_scope_name} 聚合后的训练标签仍然只有一个类别，无法训练。")
        return [], [], []

    print(f"\n[{route_scope_name} - 纯跨数据集训练启动 | Train: {aux_dataset_names} -> Test: {target_dataset_name}(100% valid)]")

    selection_cache = {}
    selection_metrics = []

    for model_type in MODELS_TO_TRY:
        try:
            final_res = train_and_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_test,
                target_map,
                model_type,
                use_smote=USE_SMOTE
            )
            selection_cache[model_type] = final_res
            selection_metrics.append({
                "Model": model_type,
                "Accuracy": final_res["acc"],
                "Train_Time_ms": final_res["train_time_ms"],
                "Pred_Latency_us": final_res["pred_latency_us"],
                "Test_Size": final_res["test_size"]
            })
            print(f"  > {model_type:<15} | 跨数据集目标域准确率: {final_res['acc']:.4%}")
        except ImportError:
            print(f"  > ⚠️ 缺少 {model_type} 引擎依赖库。")
        except Exception as e:
            print(f"  > ⚠️ {model_type} 训练失败: {e}")

    if not selection_metrics:
        print(f"❌ {route_scope_name} 没有成功训练出的模型，跳过。")
        return [], [], []

    best_model = max(selection_metrics, key=lambda x: x["Accuracy"])["Model"]
    strategy_desc = "Cross-Dataset Holdout Accuracy"

    res = selection_cache[best_model]
    candidate_set_name = " vs ".join(algo_list)
    system_acc = calculate_system_accuracy(X_test, y_global_best_test, res['classifier'], res['real_classes'], target_map)

    os.makedirs(output_dir, exist_ok=True)
    save_onnx_model(res['classifier'], best_model, X_train.shape[1], output_dir, "router.onnx", res['real_classes'])
    save_class_labels(output_dir, algo_list)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        f.write("┃         Cross-Dataset Holdout Routing Assessment Report          ┃\n")
        f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
        f.write(f"┃ 测试集  : {target_dataset_name:<15} 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} ┃\n")
        f.write(f"┃ 配置    : {config_name:<52} ┃\n")
        f.write(f"┃ 候选集  : {candidate_set_name:<52} ┃\n")
        f.write(f"┃ 训练源  : {', '.join(aux_dataset_names):<52} ┃\n")
        f.write(f"┃ 选用策略: {strategy_desc:<52} ┃\n")
        f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")

        f.write("【零 | 训练源统计 (Train Source Summary)】\n")
        f.write("-" * 80 + "\n")
        for stat in train_dataset_stats:
            f.write(
                f"  ▶ {stat['Dataset']}: valid={stat['Valid_Samples']}, "
                f"unknown={stat['Unknown_Count']}/{stat['Total_Samples']} ({stat['Unknown_Rate']:.4%})\n"
            )
        f.write(f"  ▶ 聚合训练样本数: {len(X_train)}\n")
        f.write(f"  ▶ 目标数据集测试样本数(100% valid): {len(X_test)}\n\n")

        f.write("【壹 | 测试结果 (Cross-Dataset Holdout Evaluation)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  ▶ 选定引擎 : {best_model}\n")
        f.write(f"  ▶ 候选算法 : {', '.join(algo_list)}\n")
        f.write(f"  ▶ 目标集 Unknown 样本数 : {prepared_target['unknown_count']} / {prepared_target['total_count']} ({prepared_target['unknown_rate']:.4%})\n")
        f.write(f"  🚀 目标集分类准确率: {res['acc']:.4%}\n")
        f.write(f"  🚀 目标集端到端分发准确率: {system_acc:.4%}\n\n")

        f.write("【贰 | 胜出模型透视 (Selected Model Deep Dive)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"[胜出者: {best_model}]\n")
        f.write("  ▶ 详细分类报告:\n  " + res['cls_report'].replace('\n', '\n  ') + "\n")
        f.write("  ▶ 混淆矩阵:\n  " + res['cm_df'].to_string().replace('\n', '\n  ') + "\n\n")
        f.write("  ▶ 特征重要性:\n  " + res['importance_str'].replace('\n', '\n  ') + "\n\n")

    dataset_metrics = []
    for m in selection_metrics:
        m_copy = m.copy()
        m_copy.update({
            "Mode": "CrossDatasetHoldout",
            "Dataset": target_dataset_name,
            "Train_Datasets": ",".join(aux_dataset_names),
            "Config": config_name,
            "Layer": route_scope_name,
            "Candidate_Set": candidate_set_name,
            "Unknown_Count": prepared_target["unknown_count"],
            "Unknown_Rate": prepared_target["unknown_rate"],
            "Total_Samples": prepared_target["total_count"]
        })
        dataset_metrics.append(m_copy)

    dataset_metrics.append({
        "Mode": "CrossDatasetHoldout",
        "Dataset": target_dataset_name,
        "Train_Datasets": ",".join(aux_dataset_names),
        "Config": config_name,
        "Layer": f"{route_scope_name}_System_End_to_End",
        "Model": f"CrossDataset({best_model})",
        "Accuracy": system_acc,
        "Train_Time_ms": np.nan,
        "Pred_Latency_us": np.nan,
        "Test_Size": len(y_global_best_test[y_global_best_test != 'Unknown']),
        "Candidate_Set": candidate_set_name,
        "Unknown_Count": prepared_target["unknown_count"],
        "Unknown_Rate": prepared_target["unknown_rate"],
        "Total_Samples": prepared_target["total_count"]
    })

    dataset_importances = []
    if res['importances_dict']:
        imp = res['importances_dict'].copy()
        imp.update({
            "Mode": "CrossDatasetHoldout",
            "Dataset": target_dataset_name,
            "Train_Datasets": ",".join(aux_dataset_names),
            "Config": config_name,
            "Layer": route_scope_name,
            "Model": best_model,
            "Candidate_Set": candidate_set_name
        })
        dataset_importances.append(imp)

    return dataset_metrics, [], dataset_importances

def train_shared_multi_dataset_generalization_for_candidates(
    dataset_names,
    config_name,
    algo_list,
    output_dir,
    report_path,
    route_scope_name,
    evaluation_root=None,
    evaluation_files=None,
    evaluation_label_column="Target",
    model_type=DEFAULT_ROUTER_MODEL
):
    """Train on all training samples, then evaluate only on independent queries."""
    if not dataset_names:
        print(f"❌ {route_scope_name} 没有可用的数据集。")
        return [], [], []

    target_map = {algo: idx for idx, algo in enumerate(algo_list)}
    train_feature_frames = []
    train_target_frames = []
    dataset_stats = []

    for dataset_name in dataset_names:
        df = load_dataset_dataframe(dataset_name, config_name)
        if df is None:
            return [], [], []

        prepared = prepare_routing_training_data(
            df,
            dataset_name,
            algo_list,
            f"{route_scope_name} | Full training data: {dataset_name}"
        )
        if prepared is None:
            return [], [], []

        valid_indices = prepared["valid_indices"]
        dataset_X = prepared["X"].loc[valid_indices].copy()
        dataset_y = prepared["work_df"].loc[valid_indices, "Target"].copy()

        train_feature_frames.append(dataset_X)
        train_target_frames.append(dataset_y)
        dataset_stats.append({
            "Dataset": dataset_name,
            "Train_Samples": len(valid_indices),
            "Unknown_Count": prepared["unknown_count"],
            "Unknown_Rate": prepared["unknown_rate"],
            "Total_Samples": prepared["total_count"]
        })

    X_train = pd.concat(train_feature_frames, axis=0, ignore_index=True)
    y_train = pd.concat(train_target_frames, axis=0, ignore_index=True)

    if y_train.nunique() < 2:
        print(f"❌ {route_scope_name} 聚合后的训练标签只有一个类别，无法训练。")
        return [], [], []

    if model_type not in MODELS_TO_TRY:
        print(f"❌ 预先指定的模型 {model_type} 不在支持列表中: {MODELS_TO_TRY}")
        return [], [], []

    # The model type is fixed and the model is fully fitted before any external
    # evaluation labels are loaded. External accuracy cannot affect selection.
    print(f"\n[预先固定路由模型: {model_type} | 外部评测不参与模型选择]")
    try:
        fit_result = fit_routing_model(
            X_train,
            y_train,
            target_map,
            model_type,
            use_smote=USE_SMOTE
        )
    except ImportError:
        print(f"❌ 缺少预先指定模型 {model_type} 的引擎依赖库。")
        return [], [], []
    except Exception as exc:
        print(f"❌ 预先指定模型 {model_type} 训练失败: {exc}")
        return [], [], []

    external_evaluation_sets = {}
    eval_feature_frames = []
    eval_target_frames = []
    for dataset_name in dataset_names:
        eval_df, eval_csv_path = load_external_evaluation_dataframe(
            dataset_name,
            config_name,
            evaluation_root=evaluation_root,
            evaluation_files=evaluation_files
        )
        if eval_df is None:
            return [], [], []
        try:
            prepared_eval = prepare_external_routing_evaluation_data(
                eval_df,
                dataset_name,
                algo_list,
                evaluation_label_column
            )
        except ValueError as exc:
            print(f"❌ {exc}")
            return [], [], []
        prepared_eval["csv_path"] = eval_csv_path
        external_evaluation_sets[dataset_name] = prepared_eval
        eval_feature_frames.append(prepared_eval["X"])
        eval_target_frames.append(prepared_eval["y"])

    X_eval_all = pd.concat(eval_feature_frames, axis=0, ignore_index=True)
    y_eval_all = pd.concat(eval_target_frames, axis=0, ignore_index=True)

    print(
        f"\n[{route_scope_name} | 训练数据 100% 用于拟合 | "
        f"另一批独立查询用于准确率评估]"
    )

    combined_evaluation = evaluate_trained_model(
        X_eval_all,
        y_eval_all,
        target_map,
        fit_result["classifier"],
        fit_result["real_classes"]
    )
    result = {**fit_result, **combined_evaluation}
    print(f"  > {model_type:<15} | 独立查询合并准确率: {result['acc']:.4%}")

    candidate_set_name = " vs ".join(algo_list)
    dataset_evaluations = {}

    for dataset_name in dataset_names:
        evaluation_set = external_evaluation_sets[dataset_name]
        evaluation = evaluate_trained_model(
            evaluation_set["X"],
            evaluation_set["y"],
            target_map,
            result["classifier"],
            result["real_classes"]
        )
        dataset_evaluations[dataset_name] = evaluation
        print(
            f"  > {dataset_name:<15} | 独立查询准确率: {evaluation['acc']:.4%} | "
            f"评测样本: {evaluation['test_size']}"
        )

    os.makedirs(output_dir, exist_ok=True)
    save_onnx_model(
        result["classifier"],
        model_type,
        X_train.shape[1],
        output_dir,
        "router.onnx",
        result["real_classes"]
    )
    save_class_labels(output_dir, algo_list)

    macro_accuracy = np.mean([item["acc"] for item in dataset_evaluations.values()])
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("Shared Multi-Dataset Training and Independent Evaluation Report\n")
        report.write("=" * 80 + "\n")
        report.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"配置: {config_name}\n")
        report.write(f"数据集: {', '.join(dataset_names)}\n")
        report.write(f"候选算法: {candidate_set_name}\n")
        report.write(f"预先固定模型: {model_type}\n")
        report.write("模型选择: 在读取独立评测集之前由命令行/默认配置确定\n")
        report.write("切分方式: 不切分，每个数据集 100% 有效样本用于训练\n")
        report.write("评估方式: 仅使用外部提供的另一批带正确路由标签的查询\n")
        report.write("注意: 不计算、不记录训练集拟合准确率\n\n")

        report.write("【训练样本】\n")
        report.write("-" * 80 + "\n")
        for stat in dataset_stats:
            report.write(
                f"{stat['Dataset']}: train={stat['Train_Samples']}, "
                f"unknown={stat['Unknown_Count']}/"
                f"{stat['Total_Samples']} ({stat['Unknown_Rate']:.4%})\n"
            )
        report.write(f"聚合训练样本数: {len(X_train)}\n")
        report.write(f"全局模型训练耗时: {result['train_time_ms']:.2f} ms\n")
        report.write(f"独立评测集总样本数: {len(X_eval_all)}\n")
        report.write(f"独立评测集合并准确率: {result['acc']:.4%}\n")
        report.write(f"八个数据集独立评测准确率宏平均: {macro_accuracy:.4%}\n\n")

        report.write("【各数据集独立查询评测结果】\n")
        report.write("-" * 80 + "\n")
        for dataset_name in dataset_names:
            evaluation_set = external_evaluation_sets[dataset_name]
            evaluation = dataset_evaluations[dataset_name]
            report.write(f"[{dataset_name}]\n")
            report.write(f"CSV: {evaluation_set['csv_path']}\n")
            report.write(f"Evaluation size: {evaluation['test_size']}\n")
            report.write(f"Ignored labels: {evaluation_set['ignored_count']}/{evaluation_set['total_count']}\n")
            report.write(f"Accuracy: {evaluation['acc']:.4%}\n")
            report.write("Classification report:\n" + evaluation["cls_report"] + "\n")
            report.write("Confusion matrix:\n" + evaluation["cm_df"].to_string() + "\n\n")

        report.write("【全局模型特征重要性】\n")
        report.write("-" * 80 + "\n")
        report.write(result["importance_str"] + "\n")

    train_datasets_desc = ",".join([f"{name}(100%)" for name in dataset_names])
    dataset_metrics = []
    for dataset_name in dataset_names:
        evaluation_set = external_evaluation_sets[dataset_name]
        evaluation = dataset_evaluations[dataset_name]
        common_fields = {
            "Mode": "IndependentRoutingEvaluation",
            "Dataset": dataset_name,
            "Train_Datasets": train_datasets_desc,
            "Config": config_name,
            "Candidate_Set": candidate_set_name,
            "Unknown_Count": evaluation_set["ignored_count"],
            "Unknown_Rate": evaluation_set["ignored_count"] / evaluation_set["total_count"],
            "Total_Samples": evaluation_set["total_count"],
            "Evaluation_Scope": "IndependentQueries",
            "Evaluation_CSV": evaluation_set["csv_path"],
            "Model_Selection": "PreFixedBeforeEvaluation"
        }
        dataset_metrics.append({
            **common_fields,
            "Layer": route_scope_name,
            "Model": model_type,
            "Accuracy": evaluation["acc"],
            "Train_Time_ms": result["train_time_ms"],
            "Pred_Latency_us": evaluation["pred_latency_us"],
            "Train_Size": len(X_train),
            "Evaluation_Size": evaluation["test_size"]
        })

    dataset_importances = []
    if result["importances_dict"]:
        importance = result["importances_dict"].copy()
        importance.update({
            "Mode": "IndependentRoutingEvaluation",
            "Dataset": "AllDatasets",
            "Train_Datasets": train_datasets_desc,
            "Config": config_name,
            "Layer": route_scope_name,
            "Model": model_type,
            "Candidate_Set": candidate_set_name
        })
        dataset_importances.append(importance)

    print(f"\n✅ 全量训练和八数据集独立查询评测完成: {report_path}")
    return dataset_metrics, [], dataset_importances

def process_multi_dataset_generalization(
    config_name,
    dataset_names=None,
    train_full_model=True,
    train_pairwise_models=True,
    evaluation_root=None,
    evaluation_files=None,
    evaluation_label_column="Target",
    model_type=DEFAULT_ROUTER_MODEL
):
    dataset_names = list(dataset_names or DATASET_LIST)
    print(f"\n{'='*70}")
    print(
        f"🌍 开始共享全局路由训练: "
        f"每个数据集 100% 有效样本用于训练 | 配置: {config_name}"
    )
    print(f"{'='*70}")

    if not dataset_names:
        print("❌ 没有配置可用数据集。")
        return [], [], []

    output_dir = os.path.join(
        SELECT_MODELS_ROOT_DIR,
        "global",
        SELECT_MODELS_RUN_DIR,
        config_name,
        "generalization_8datasets"
    )
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    all_importances = []

    full_algo_list = select_common_algorithms_for_datasets(dataset_names, config_name)
    if full_algo_list is None:
        return [], [], []

    if train_full_model and len(full_algo_list) >= 2:
        full_report_path = os.path.join(output_dir, "SmartRoute_8Datasets_Global_Report.txt")
        full_metrics, _, full_importances = train_shared_multi_dataset_generalization_for_candidates(
            dataset_names=dataset_names,
            config_name=config_name,
            algo_list=full_algo_list,
            output_dir=output_dir,
            report_path=full_report_path,
            route_scope_name="Shared MultiDataset Full Training (Full Candidate Set)",
            evaluation_root=evaluation_root,
            evaluation_files=evaluation_files,
            evaluation_label_column=evaluation_label_column,
            model_type=model_type
        )
        all_metrics.extend(full_metrics)
        all_importances.extend(full_importances)
    elif train_full_model:
        print("⚠️ 所有数据集之间没有至少两个共同候选算法，跳过 full model。")

    if train_pairwise_models and len(full_algo_list) >= 2:
        for pair_algos in combinations(full_algo_list, 2):
            pair_list = list(pair_algos)
            pair_name = build_candidate_set_name(pair_list)
            pair_output_dir = os.path.join(output_dir, "pairwise_models", pair_name)
            pair_report_path = os.path.join(
                pair_output_dir,
                f"SmartRoute_8Datasets_Global_Pair_Report_{pair_name}.txt"
            )
            pair_scope_name = f"Shared MultiDataset Full Training Pairwise ({pair_list[0]} vs {pair_list[1]})"
            pair_metrics, _, pair_importances = train_shared_multi_dataset_generalization_for_candidates(
                dataset_names=dataset_names,
                config_name=config_name,
                algo_list=pair_list,
                output_dir=pair_output_dir,
                report_path=pair_report_path,
                route_scope_name=pair_scope_name,
                evaluation_root=evaluation_root,
                evaluation_files=evaluation_files,
                evaluation_label_column=evaluation_label_column,
                model_type=model_type
            )
            all_metrics.extend(pair_metrics)
            all_importances.extend(pair_importances)

    return all_metrics, [], all_importances

def process_cross_dataset_holdout(
    target_dataset_name,
    config_name,
    train_full_model=True,
    train_pairwise_models=True
):
    print(f"\n{'='*70}")
    print(f"🌍 开始 7 训 1 测评估: Train(其他 7 个数据集) -> Test({target_dataset_name}) | 配置: {config_name}")
    print(f"{'='*70}")

    aux_dataset_names = [name for name in DATASET_LIST if name != target_dataset_name]
    if not aux_dataset_names:
        print(f"❌ 无法进行 target={target_dataset_name} 的 7 训 1 测：没有剩余训练数据集。")
        return [], [], []

    output_dir = os.path.join(
        LEAVE1_SELECT_MODELS_ROOT_DIR,
        target_dataset_name,
        SELECT_MODELS_RUN_DIR,
        config_name,
        "cross_dataset_holdout"
    )
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    all_importances = []

    full_algo_list = select_common_algorithms(aux_dataset_names, target_dataset_name, config_name)
    if full_algo_list is None:
        return [], [], []

    if train_full_model and len(full_algo_list) >= 2:
        full_report_path = os.path.join(output_dir, f"SmartRoute_CrossDataset_Report_{target_dataset_name}.txt")
        full_metrics, _, full_importances = train_cross_dataset_holdout_for_candidates(
            target_dataset_name=target_dataset_name,
            aux_dataset_names=aux_dataset_names,
            config_name=config_name,
            algo_list=full_algo_list,
            output_dir=output_dir,
            report_path=full_report_path,
            route_scope_name="CrossDataset Holdout (Full Candidate Set)"
        )
        all_metrics.extend(full_metrics)
        all_importances.extend(full_importances)
    elif train_full_model:
        print(f"⚠️ {target_dataset_name} 与其余 7 个数据集没有共同的完整候选算法集合，跳过 full model。")

    if train_pairwise_models and len(full_algo_list) >= 2:
        for pair_algos in combinations(full_algo_list, 2):
            pair_list = list(pair_algos)
            pair_name = build_candidate_set_name(pair_list)
            pair_output_dir = os.path.join(output_dir, "pairwise_models", pair_name)
            pair_report_path = os.path.join(pair_output_dir, f"SmartRoute_CrossDataset_Pair_Report_{target_dataset_name}_{pair_name}.txt")
            pair_scope_name = f"CrossDataset Pairwise ({pair_list[0]} vs {pair_list[1]})"
            pair_metrics, _, pair_importances = train_cross_dataset_holdout_for_candidates(
                target_dataset_name=target_dataset_name,
                aux_dataset_names=aux_dataset_names,
                config_name=config_name,
                algo_list=pair_list,
                output_dir=pair_output_dir,
                report_path=pair_report_path,
                route_scope_name=pair_scope_name
            )
            all_metrics.extend(pair_metrics)
            all_importances.extend(pair_importances)

    return all_metrics, [], all_importances

def discover_configs():
    if not os.path.exists(EDA_ROOT_DIR):
        return []
    return sorted([
        name for name in os.listdir(EDA_ROOT_DIR)
        if os.path.isdir(os.path.join(EDA_ROOT_DIR, name))
    ])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SODA selector models for one or more routing config groups."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=SUPPORTED_ROUTING_CONFIGS,
        help="Optional routing config groups to train. Example: --configs FAVOR ACORN-gamma",
    )
    parser.add_argument(
        "--pairwise-only",
        action="store_true",
        help="Only train pairwise models for every 2-of-N candidate combination.",
    )
    parser.add_argument(
        "--include-full-model",
        action="store_true",
        help="Train the original full multi-class model in addition to pairwise models.",
    )
    parser.add_argument(
        "--full-model-only",
        action="store_true",
        help="Only train the original full multi-class model, for example FAVOR / UNG+ / pre-filter, without pairwise models.",
    )
    parser.add_argument(
        "--eval-data-root",
        default=os.environ.get("ALPS_ROUTING_EVAL_DIR"),
        help=(
            "Root directory containing independent routing-evaluation CSVs. "
            "Files may be arranged as ROOT/CONFIG/DATASET/DATASET_routing_eval.csv."
        ),
    )
    parser.add_argument(
        "--eval-files",
        nargs="+",
        metavar="DATASET=CSV",
        help="Explicit independent evaluation CSV for each dataset; overrides --eval-data-root.",
    )
    parser.add_argument(
        "--eval-label-column",
        default="Target",
        help="Column containing the correct route name (or its class index). Default: Target.",
    )
    parser.add_argument(
        "--model-type",
        choices=MODELS_TO_TRY,
        default=DEFAULT_ROUTER_MODEL,
        help=(
            "Router model fixed before loading independent evaluation data. "
            f"Default: {DEFAULT_ROUTER_MODEL}."
        ),
    )
    return parser.parse_args()

def parse_evaluation_file_mapping(items):
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"评测文件参数格式错误: '{item}'，应为 DATASET=/path/to/file.csv")
        dataset_name, csv_path = item.split("=", 1)
        dataset_name = dataset_name.strip()
        csv_path = csv_path.strip()
        if dataset_name not in DATASET_LIST:
            raise ValueError(f"未知数据集 '{dataset_name}'，可选值: {DATASET_LIST}")
        if not csv_path:
            raise ValueError(f"{dataset_name} 的评测 CSV 路径为空。")
        if dataset_name in mapping:
            raise ValueError(f"数据集 {dataset_name} 的评测 CSV 被重复指定。")
        mapping[dataset_name] = csv_path
    return mapping

def resolve_config_names(selected_configs=None):
    discovered_configs = discover_configs()
    target_configs = selected_configs if selected_configs else AVAILABLE_ROUTING_CONFIGS

    unsupported_configs = [cfg for cfg in target_configs if cfg not in SUPPORTED_ROUTING_CONFIGS]
    if unsupported_configs:
        print(f"⚠️ 以下配置不在支持列表中，将跳过: {', '.join(unsupported_configs)}")

    missing_configs = [cfg for cfg in target_configs if cfg not in discovered_configs]
    if missing_configs:
        print(f"⚠️ 以下配置目录不存在，将跳过: {', '.join(missing_configs)}")

    return [
        cfg for cfg in target_configs
        if cfg in SUPPORTED_ROUTING_CONFIGS and cfg in discovered_configs
    ]

def main():
    args = parse_args()
    try:
        evaluation_files = parse_evaluation_file_mapping(args.eval_files)
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    if RUN_MULTI_DATASET_GENERALIZATION and not args.eval_data_root and not evaluation_files:
        print(
            "❌ 全量共享训练需要另一批独立查询进行准确率评估。\n"
            "   请使用 --eval-data-root ROOT，或用 --eval-files Dataset=/path/file.csv 分别提供。"
        )
        return

    config_names = resolve_config_names(args.configs)
    if not config_names:
        print(f"❌ 在 {EDA_ROOT_DIR} 下未发现可训练的宽表配置目录。")
        return

    if args.pairwise_only and args.full_model_only:
        print("❌ 参数冲突: --pairwise-only 与 --full-model-only 不能同时使用。")
        return

    pairwise_only = DEFAULT_PAIRWISE_ONLY
    if args.pairwise_only:
        pairwise_only = True
    if args.include_full_model:
        pairwise_only = False

    full_model_only = DEFAULT_FULL_MODEL_ONLY
    if args.full_model_only:
        full_model_only = True

    train_pairwise_models = not full_model_only

    if full_model_only:
        print("🧭 当前模式: 仅训练原始三分类模型（如 FAVOR / UNG+ / pre-filter）")
    elif pairwise_only:
        print("🧭 当前模式: 仅训练 pairwise 模型")
    else:
        print("🧭 当前模式: 同时训练原始三分类模型和 pairwise 模型")

    if RUN_SINGLE_DATASET_TRAINING:
        print("📦 已启用单数据集训练模式")
    if RUN_MULTI_DATASET_GENERALIZATION:
        print(f"🌍 已启用 8 数据集全量共享路由训练: {', '.join(GENERALIZATION_TARGET_DATASETS)}")
    if RUN_CROSS_DATASET_HOLDOUT:
        print(f"🧪 已启用 7 训 1 测模式，目标数据集: {', '.join(GENERALIZATION_HOLDOUT_TARGET_DATASETS)}")

    for config_name in config_names:
        print(f"\n{'#' * 80}")
        print(f"📦 开始训练配置: {config_name}")
        print(f"{'#' * 80}")

        standard_metrics = []
        standard_ablation = []
        standard_importances = []
        leave1_metrics = []
        leave1_ablation = []
        leave1_importances = []
        
        if RUN_SINGLE_DATASET_TRAINING:
            for dataset in DATASET_LIST:
                res = process_single_dataset(
                    dataset,
                    config_name,
                    train_full_model=not pairwise_only,
                    train_pairwise_models=train_pairwise_models
                )
                if res:
                    d_metrics, d_ablation, d_importances = res
                    standard_metrics.extend(d_metrics)
                    standard_ablation.extend(d_ablation)
                    standard_importances.extend(d_importances)

        if RUN_MULTI_DATASET_GENERALIZATION:
            res = process_multi_dataset_generalization(
                config_name,
                dataset_names=GENERALIZATION_TARGET_DATASETS,
                train_full_model=not pairwise_only,
                train_pairwise_models=train_pairwise_models,
                evaluation_root=args.eval_data_root,
                evaluation_files=evaluation_files,
                evaluation_label_column=args.eval_label_column,
                model_type=args.model_type
            )
            if res:
                d_metrics, d_ablation, d_importances = res
                standard_metrics.extend(d_metrics)
                standard_ablation.extend(d_ablation)
                standard_importances.extend(d_importances)

        if RUN_CROSS_DATASET_HOLDOUT:
            for target_dataset in GENERALIZATION_HOLDOUT_TARGET_DATASETS:
                res = process_cross_dataset_holdout(
                    target_dataset,
                    config_name,
                    train_full_model=not pairwise_only,
                    train_pairwise_models=train_pairwise_models
                )
                if res:
                    d_metrics, d_ablation, d_importances = res
                    leave1_metrics.extend(d_metrics)
                    leave1_ablation.extend(d_ablation)
                    leave1_importances.extend(d_importances)

        if standard_metrics:
            write_summary_outputs(
                standard_metrics,
                standard_ablation,
                standard_importances,
                build_summary_output_dir(config_name)
            )

        if leave1_metrics:
            write_summary_outputs(
                leave1_metrics,
                leave1_ablation,
                leave1_importances,
                build_leave1_summary_output_dir(config_name)
            )

if __name__ == "__main__":
    main()
