"""
書誌レコード一致判定のコア関数モジュール
協力者に提供するための関数として使用可能
"""
import os
import time
import json
import csv
import re
import pickle
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from jellyfish import jaro_winkler_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================================
# コア判定ロジック関数群
# ============================================================================

def normalize_publisher(publisher_str):
    """出版社名を正規化する"""
    if not publisher_str:
        return ""
    if ':' in publisher_str:
        return publisher_str.split(':')[1].strip()
    return publisher_str.strip()


def extract_series_info(title):
    """シリーズ名と巻数を抽出する"""
    if not title:
        return None, None
        
    # シリーズ + 数字のパターンを検出
    series_pattern = re.compile(r'(.+?シリーズ)\s*(\d+)')
    match = series_pattern.search(title)
    
    if match:
        return match.group(1), int(match.group(2))
    
    # その他のシリーズパターン
    if 'シリーズ' in title:
        return title.split('シリーズ')[0] + 'シリーズ', None
        
    return None, None


def compute_name_similarity(author1, author2):
    """著者名の類似度を計算"""
    if not author1 or not author2:
        return 0.0
        
    # 著者表記の正規化（「/著」などを削除）
    author1 = re.sub(r'/.*$', '', author1).strip()
    author2 = re.sub(r'/.*$', '', author2).strip()
    
    return jaro_winkler_similarity(author1, author2)


def compute_keyword_overlap(title1, title2):
    """タイトル間のキーワード重複率を計算"""
    if not title1 or not title2:
        return 0.0
        
    # ストップワード除去
    stop_words = ['の', 'に', 'は', 'を', 'と', 'が', 'から', 'より', 'による']
    
    # 単語抽出（簡易的）
    def extract_keywords(title):
        # 記号を空白に置換
        cleaned = re.sub(r'[「」『』（）\(\)\[\]\{\}]', ' ', title)
        words = [w for w in cleaned.split() if w not in stop_words and len(w) > 1]
        return set(words)
    
    keywords1 = extract_keywords(title1)
    keywords2 = extract_keywords(title2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Jaccard係数
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0


def extract_features(record1, record2):
    """2つのレコードから特徴量を抽出"""
    features = {}
    
    # 1. 基本的な文字列類似度
    features['title_similarity'] = jaro_winkler_similarity(
        record1.get('title', ''), record2.get('title', ''))
    
    # 2. 出版情報の一致
    features['same_publisher'] = int(
        normalize_publisher(record1.get('publisher', '')) == 
        normalize_publisher(record2.get('publisher', '')))
    
    features['same_pubdate'] = int(
        record1.get('pubdate', '') == record2.get('pubdate', ''))
    
    # 3. 著者類似度
    features['author_similarity'] = compute_name_similarity(
        record1.get('author', ''), record2.get('author', ''))
    
    # 4. シリーズ情報
    series1, vol1 = extract_series_info(record1.get('title', ''))
    series2, vol2 = extract_series_info(record2.get('title', ''))
    
    features['same_series'] = int(bool(series1 and series2 and series1 == series2))
    features['same_volume'] = int(vol1 == vol2 and vol1 is not None)
    
    # 5. キーワード重複
    features['keyword_overlap'] = compute_keyword_overlap(
        record1.get('title', ''), record2.get('title', ''))
        
    # 6. タイトルの長さの差（正規化）
    len1 = len(record1.get('title', ''))
    len2 = len(record2.get('title', ''))
    if len1 > 0 and len2 > 0:
        features['title_length_diff'] = abs(len1 - len2) / max(len1, len2)
    else:
        features['title_length_diff'] = 1.0
    
    # 7. 欠損データの状況
    features['title_missing'] = int(not bool(record1.get('title', '')) or not bool(record2.get('title', '')))
    features['author_missing'] = int(not bool(record1.get('author', '')) or not bool(record2.get('author', '')))
    features['publisher_missing'] = int(not bool(record1.get('publisher', '')) or not bool(record2.get('publisher', '')))
    features['pubdate_missing'] = int(not bool(record1.get('pubdate', '')) or not bool(record2.get('pubdate', '')))
    
    return features


def string_similarity_prediction(record1, record2, threshold=0.85):
    """単純な文字列類似度による予測"""
    title1 = record1.get('title', '')
    title2 = record2.get('title', '')
    
    if not title1 or not title2:
        # タイトルがない場合は他のフィールドで判断
        if (record1.get('author') == record2.get('author') and 
            normalize_publisher(record1.get('publisher', '')) == normalize_publisher(record2.get('publisher', '')) and
            record1.get('pubdate') == record2.get('pubdate') and
            all([record1.get(f) for f in ['author', 'publisher', 'pubdate']]) and
            all([record2.get(f) for f in ['author', 'publisher', 'pubdate']])):
            return 1, 0.95
        return 0, 0.1
    
    similarity = jaro_winkler_similarity(title1, title2)
    prediction = 1 if similarity >= threshold else 0
    confidence = similarity if prediction == 1 else 1 - similarity
    
    return prediction, confidence


def hybrid_prediction(record1, record2, ml_model, feature_names, threshold=0.85):
    """機械学習と文字列類似度を状況に応じて使い分ける予測関数"""
    
    # データの欠損状況を確認
    missing_data = any([
        not record1.get('author'), not record2.get('author'),
        not record1.get('publisher'), not record2.get('publisher'),
        not record1.get('pubdate'), not record2.get('pubdate')
    ])
    
    # タイトルの有無を確認
    title_present = bool(record1.get('title')) and bool(record2.get('title'))
    
    # 文字列類似度の計算
    title_similarity = 0
    if title_present:
        title_similarity = jaro_winkler_similarity(record1.get('title', ''), record2.get('title', ''))
    
    # 特別なケース：タイトル以外の情報が非常に類似（シリーズものなど）
    author_match = compute_name_similarity(record1.get('author', ''), record2.get('author', '')) > 0.9
    publisher_match = normalize_publisher(record1.get('publisher', '')) == normalize_publisher(record2.get('publisher', ''))
    date_match = record1.get('pubdate') == record2.get('pubdate')
    
    # 判断ロジック
    if missing_data or not title_present:
        # データ不足の場合はシンプルな文字列類似度または他フィールドの一致で判断
        return string_similarity_prediction(record1, record2, threshold)
    
    # 十分なデータがある場合は機械学習モデルを使用
    features_dict = extract_features(record1, record2)
    features_values = [features_dict[name] for name in feature_names]
    
    ml_prediction = ml_model.predict([features_values])[0]
    ml_confidence = ml_model.predict_proba([features_values])[0][1]
    
    # 特定条件では機械学習の結果を上書き
    if ml_prediction == 0:
        # 機械学習が「不一致」と判断した場合の特別ルール
        
        # ルール1: タイトルは異なるが他の全情報が一致（シリーズものの可能性）
        if author_match and publisher_match and date_match:
            # シリーズ情報を確認
            series1, _ = extract_series_info(record1.get('title', ''))
            series2, _ = extract_series_info(record2.get('title', ''))
            if series1 and series2 and series1 == series2:
                return 1, 0.9
                
        # ルール2: タイトルの類似度が非常に高い場合
        if title_similarity > 0.615:
            return 1, title_similarity
            
    # 上記の特別ルールに当てはまらない場合は機械学習の結果を使用
    return ml_prediction, ml_confidence


def save_model(model, feature_names, filename):
    """モデルと特徴量名を保存"""
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"モデルを{filename}に保存しました。")


def load_model(filename):
    """保存したモデルと特徴量名を読み込む"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']


# ============================================================================
# API/関数用のラッパー関数群
# ============================================================================

# グローバル変数でモデルをキャッシュ
_model_cache: Optional[Tuple[Any, list]] = None
_model_file: Optional[str] = None


def get_model(model_file: Optional[str] = None) -> Tuple[Any, list]:
    """
    モデルを読み込む（キャッシュ機能付き）
    
    Args:
        model_file: モデルファイルのパス。Noneの場合はデフォルトパスを使用
        
    Returns:
        (model, feature_names) のタプル
    """
    global _model_cache, _model_file
    
    # デフォルトのモデルファイルパス
    if model_file is None:
        model_file = os.path.join(os.path.dirname(__file__), 'bibliography_matcher.pkl')
    
    # すでに読み込まれていて、同じファイルの場合はキャッシュを返す
    if _model_cache is not None and _model_file == model_file:
        return _model_cache
    
    # モデルを読み込む
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_file}")
    
    model, feature_names = load_model(model_file)
    _model_cache = (model, feature_names)
    _model_file = model_file
    
    return model, feature_names


def match_records(
    record1: Dict[str, str],
    record2: Dict[str, str],
    model_file: Optional[str] = None,
    threshold: float = 0.85,
    measure_time: bool = False
) -> Dict[str, Any]:
    """
    2つの書誌レコードが一致するか判定する
    
    Args:
        record1: 1つ目のレコード（title, author, publisher, pubdate を含む辞書）
        record2: 2つ目のレコード（title, author, publisher, pubdate を含む辞書）
        model_file: モデルファイルのパス（オプション）
        threshold: 文字列類似度の閾値（デフォルト: 0.85）
        measure_time: 処理時間を計測するか（デフォルト: False）
        
    Returns:
        判定結果の辞書:
        {
            "label": 0 or 1,  # 0=不一致, 1=一致
            "confidence": float,  # 確信度 (0.0-1.0)
            "method": str,  # 使用した手法
            "processing_time_ms": float  # 処理時間（ミリ秒、measure_time=Trueの場合）
        }
        
    Example:
        >>> record1 = {
        ...     "title": "絵画における真理 下 叢書・ウニベルシタス 591",
        ...     "author": "ジャック・デリダ/〔著〕",
        ...     "publisher": "東京:法政大学出版局",
        ...     "pubdate": "1998.07"
        ... }
        >>> record2 = {
        ...     "title": "絵画における真理 下  [叢書・ウニ] (591)",
        ...     "author": "ジャック・デリダ/[著]",
        ...     "publisher": "法政大学出版局",
        ...     "pubdate": "1998.7"
        ... }
        >>> result = match_records(record1, record2)
        >>> print(result["label"])  # 1 (一致)
    """
    start_time = time.time() if measure_time else None
    
    try:
        # モデルの読み込み
        model, feature_names = get_model(model_file)
        
        # データの欠損状況を確認
        missing_data = any([
            not record1.get('author'), not record2.get('author'),
            not record1.get('publisher'), not record2.get('publisher'),
            not record1.get('pubdate'), not record2.get('pubdate')
        ])
        
        title_present = bool(record1.get('title')) and bool(record2.get('title'))
        
        # 各手法での予測
        ml_features = extract_features(record1, record2)
        ml_features_list = [ml_features[name] for name in feature_names]
        ml_pred = model.predict([ml_features_list])[0]
        ml_conf = model.predict_proba([ml_features_list])[0][1]
        
        str_pred, str_conf = string_similarity_prediction(record1, record2, threshold)
        
        # ハイブリッド予測
        hybrid_pred, hybrid_conf = hybrid_prediction(record1, record2, model, feature_names, threshold)
        
        # 使用した手法の決定
        if missing_data or not title_present:
            method_used = "string_similarity"
        elif hybrid_pred != ml_pred:
            method_used = "hybrid_rules"
        else:
            method_used = "machine_learning"
        
        result = {
            "label": int(hybrid_pred),
            "confidence": float(hybrid_conf),
            "method": method_used,
            "ml_prediction": int(ml_pred),
            "ml_confidence": float(ml_conf),
            "str_prediction": int(str_pred),
            "str_confidence": float(str_conf)
        }
        
        # 処理時間を追加
        if measure_time and start_time:
            result["processing_time_ms"] = (time.time() - start_time) * 1000
        
        return result
        
    except Exception as e:
        error_result = {
            "label": 0,
            "confidence": 0.0,
            "method": "error",
            "ml_prediction": 0,
            "ml_confidence": 0.0,
            "str_prediction": 0,
            "str_confidence": 0.0,
            "error": str(e)
        }
        if measure_time and start_time:
            error_result["processing_time_ms"] = (time.time() - start_time) * 1000
        return error_result


def match_records_batch(
    record_pairs: list,
    model_file: Optional[str] = None,
    threshold: float = 0.85,
    measure_time: bool = False
) -> Dict[str, Any]:
    """
    複数のレコードペアを一括で判定する
    
    Args:
        record_pairs: レコードペアのリスト。各要素は [record1, record2] または [record1, record2, id]
        model_file: モデルファイルのパス（オプション）
        threshold: 文字列類似度の閾値（デフォルト: 0.85）
        measure_time: 処理時間を計測するか（デフォルト: False）
        
    Returns:
        判定結果の辞書:
        {
            "results": [
                {
                    "id": str or None,
                    "label": 0 or 1,
                    "confidence": float,
                    "method": str,
                    ...
                },
                ...
            ],
            "total_count": int,
            "total_processing_time_ms": float,  # measure_time=Trueの場合
            "average_processing_time_ms": float  # measure_time=Trueの場合
        }
    """
    start_time = time.time() if measure_time else None
    
    results = []
    for pair in record_pairs:
        if len(pair) >= 2:
            record1, record2 = pair[0], pair[1]
            record_id = pair[2] if len(pair) >= 3 else None
            
            result = match_records(record1, record2, model_file, threshold, measure_time=False)
            
            if record_id is not None:
                result["id"] = record_id
            
            results.append(result)
    
    response = {
        "results": results,
        "total_count": len(results)
    }
    
    if measure_time and start_time:
        total_time = (time.time() - start_time) * 1000
        response["total_processing_time_ms"] = total_time
        response["average_processing_time_ms"] = total_time / len(results) if results else 0.0
    
    return response


# ============================================================================
# モデル訓練用関数（オプション）
# ============================================================================

def train_model_with_labels(data_file, label_file):
    """正解ラベルを使ってモデルを訓練する"""
    # データの読み込み
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 正解ラベルの読み込み
    labels_df = pd.read_csv(label_file)
    labels_dict = dict(zip(labels_df['id'], labels_df['label']))
    
    # 特徴量とラベルの準備
    feature_dicts = []
    X = []
    y = []
    record_ids = []
    record_pairs = []
    
    for group in data["data"]:
        if len(group) >= 3 and isinstance(group[0], dict) and isinstance(group[1], dict):
            record_id = group[2]
            if record_id in labels_dict:
                features_dict = extract_features(group[0], group[1])
                feature_dicts.append(features_dict)
                X.append(list(features_dict.values()))
                y.append(labels_dict[record_id])
                record_ids.append(record_id)
                record_pairs.append((group[0], group[1]))
    
    # 特徴量名
    feature_names = list(feature_dicts[0].keys())
    
    # データをnumpy配列に変換
    X = np.array(X)
    y = np.array(y)
    
    # 訓練/テスト分割
    X_train, X_test, y_train, y_test, ids_train, ids_test, pairs_train, pairs_test = train_test_split(
        X, y, record_ids, record_pairs, test_size=0.2, random_state=42, stratify=y)
    
    # モデルの訓練
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 基本的な性能評価
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"訓練精度: {train_accuracy:.4f}")
    print(f"テスト精度: {test_accuracy:.4f}")
    
    # クロスバリデーション
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5分割交差検証スコア: {cv_scores}")
    print(f"平均CV精度: {cv_scores.mean():.4f}")
    
    # テストセットでの詳細評価
    y_pred = model.predict(X_test)
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred))
    
    print("\n混同行列:")
    print(confusion_matrix(y_test, y_pred))
    
    # 特徴量の重要度
    print("\n特徴量の重要度:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # ハイブリッドアプローチの評価
    print("\nハイブリッドアプローチの評価:")
    hybrid_correct = 0
    string_sim_correct = 0
    
    for i, (X_i, y_i, pair) in enumerate(zip(X_test, y_test, pairs_test)):
        # 機械学習の予測
        ml_pred = model.predict([X_i])[0]
        
        # 文字列類似度の予測
        str_pred, _ = string_similarity_prediction(pair[0], pair[1])
        
        # ハイブリッド予測
        hybrid_pred, _ = hybrid_prediction(pair[0], pair[1], model, feature_names)
        
        # 正解数のカウント
        if ml_pred == y_i:
            hybrid_correct += 1
        
        if str_pred == y_i:
            string_sim_correct += 1
    
    print(f"機械学習のみの精度: {test_accuracy:.4f}")
    print(f"文字列類似度のみの精度: {string_sim_correct/len(y_test):.4f}")
    print(f"ハイブリッドアプローチの精度: {hybrid_correct/len(y_test):.4f}")
    
    # 誤判定の分析
    print("\n機械学習モデルによる誤判定の分析:")
    misclassified = np.where(y_test != y_pred)[0]
    for idx in misclassified[:10]:  # 最初の10件のみ表示
        record_id = ids_test[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        confidence = model.predict_proba([X_test[idx]])[0][1]
        
        # ハイブリッド予測
        hybrid_pred, hybrid_conf = hybrid_prediction(pairs_test[idx][0], pairs_test[idx][1], model, feature_names)
        
        # データを表示
        print(f"ID: {record_id}")
        print(f"正解: {true_label}, ML予測: {pred_label}, ML確信度: {confidence:.4f}")
        print(f"ハイブリッド予測: {hybrid_pred}, 確信度: {hybrid_conf:.4f}")
        print(f"レコード1: {pairs_test[idx][0]}")
        print(f"レコード2: {pairs_test[idx][1]}")
        print()
    
    return model, feature_names
