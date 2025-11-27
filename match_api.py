"""
書誌レコード一致判定のコア関数モジュール

MIT License

See the LICENSE file in the project root for license information.

Copyright (c) 2025 kota-okayama

"""
import os
import time
import json
import re
import pickle
import unicodedata
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

def normalize_string(text: str, preserve_spaces: bool = True) -> str:
    """
    文字列を包括的に正規化する
    
    Args:
        text: 正規化する文字列
        preserve_spaces: 空白を保持するか（Falseの場合は削除）
        
    Returns:
        正規化された文字列
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unicode正規化（NFKC: 互換分解後に正準合成）
    # 全角英数字を半角に、全角記号を半角に変換
    text = unicodedata.normalize('NFKC', text)
    
    # 制御文字を削除
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
    
    # 空白の正規化
    if preserve_spaces:
        # 全角空白、タブ、改行を半角空白に統一
        text = re.sub(r'[\u3000\t\n\r]+', ' ', text)
        # 連続する空白を1つに
        text = re.sub(r' +', ' ', text)
        # 前後の空白を削除
        text = text.strip()
    else:
        # すべての空白を削除
        text = re.sub(r'\s+', '', text)
    
    return text


def normalize_author_name(author_str: str) -> str:
    """
    著者名を正規化する（役割表記を除去、空白を統一）
    
    Args:
        author_str: 著者名文字列
        
    Returns:
        正規化された著者名
    """
    if not author_str:
        return ""
    
    # 基本的な正規化
    author = normalize_string(author_str, preserve_spaces=True)
    
    # 役割表記を除去（様々なパターンに対応）
    # /著, /編, /訳, 〔著〕, [著], （著）, 著, 編, 訳 など
    role_patterns = [
        r'[/／].*$',  # /以降を削除
        r'[（(〔\[【].*[著編訳監修].*[）)〕\]】]',  # 括弧内の役割表記
        r'[著編訳監修]$',  # 末尾の役割表記
        r'^\s*[著編訳監修]\s*',  # 先頭の役割表記
    ]
    
    for pattern in role_patterns:
        author = re.sub(pattern, '', author)
    
    # 再度正規化（空白の整理）
    author = normalize_string(author, preserve_spaces=True)
    
    return author


def normalize_publisher(publisher_str):
    """
    出版社名を正規化する
    
    様々な区切り文字（:, ：, 全角コロンなど）に対応
    """
    if not publisher_str:
        return ""
    
    # 基本的な正規化
    publisher = normalize_string(publisher_str, preserve_spaces=True)
    
    # 区切り文字で分割（:, ：, 全角コロンなど）
    separators = [':', '：', '，', ',', '、']
    for sep in separators:
        if sep in publisher:
            # 区切り文字以降を取得
            parts = publisher.split(sep, 1)
            if len(parts) > 1:
                publisher = parts[1].strip()
                break
    
    # 再度正規化
    publisher = normalize_string(publisher, preserve_spaces=True)
    
    return publisher


def extract_series_info(title):
    """
    シリーズ名と巻数を抽出する
    
    様々な巻数表記に対応：
    - シリーズ名 + 数字
    - 第X巻, Vol.X, 第X冊 など
    - 上/中/下 などの巻表記
    """
    if not title:
        return None, None
    
    # タイトルを正規化（比較用に空白を保持）
    normalized_title = normalize_string(title, preserve_spaces=True)
    
    # パターン1: シリーズ名 + 数字（全角・半角数字に対応）
    series_patterns = [
        r'(.+?シリーズ)\s*[（(]?\s*(\d+)\s*[）)]?',  # シリーズ名 + 数字
        r'(.+?シリーズ)\s*第\s*(\d+)\s*[巻冊]',  # シリーズ名 + 第X巻
        r'(.+?シリーズ)\s*Vol\.?\s*(\d+)',  # シリーズ名 + Vol.X
    ]
    
    for pattern in series_patterns:
        match = re.search(pattern, normalized_title)
        if match:
            series_name = normalize_string(match.group(1), preserve_spaces=True)
            try:
                volume = int(match.group(2))
                return series_name, volume
            except ValueError:
                return series_name, None
    
    # パターン2: シリーズ名のみ（巻数なし）
    if 'シリーズ' in normalized_title:
        # シリーズ名の部分を抽出
        series_match = re.search(r'(.+?シリーズ)', normalized_title)
        if series_match:
            series_name = normalize_string(series_match.group(1), preserve_spaces=True)
            return series_name, None
    
    # パターン3: 巻表記（上/中/下など）
    volume_patterns = [
        r'(.+?)\s*[（(]?\s*(上|中|下|上巻|中巻|下巻)\s*[）)]?',
        r'(.+?)\s*第\s*(\d+)\s*[巻冊]',
        r'(.+?)\s*Vol\.?\s*(\d+)',
    ]
    
    for pattern in volume_patterns:
        match = re.search(pattern, normalized_title)
        if match:
            # シリーズ名として扱う（巻数は別途処理）
            series_name = normalize_string(match.group(1), preserve_spaces=True)
            if len(match.groups()) > 1:
                vol_str = match.group(2)
                # 上/中/下を数値に変換
                vol_map = {'上': 1, '中': 2, '下': 3, '上巻': 1, '中巻': 2, '下巻': 3}
                if vol_str in vol_map:
                    return series_name, vol_map[vol_str]
                try:
                    return series_name, int(vol_str)
                except ValueError:
                    pass
            return series_name, None
        
    return None, None


def extract_edition_info(title: str) -> Optional[str]:
    """
    タイトルから版情報を抽出する
    
    Args:
        title: タイトル文字列
        
    Returns:
        版情報（文庫版、新書版、電子版、改訂版など）、見つからない場合はNone
    """
    if not title:
        return None
    
    # タイトルを正規化
    normalized_title = normalize_string(title, preserve_spaces=True)
    
    # 版情報のパターン（括弧内、括弧外の両方に対応）
    edition_patterns = [
        # 日本語の版表記
        r'[（(]?\s*(文庫版|新書版|電子版|改訂版|初版|第\d+版|増補版|新版|普及版|限定版|特装版|ハードカバー|ペーパーバック|ソフトカバー)\s*[）)]?',
        r'[（(]?\s*(文庫|新書|電子|改訂|増補|新版|普及|限定|特装)\s*[）)]?',
        # 英語の版表記
        r'[（(]?\s*(hardcover|paperback|ebook|e-book|digital|revised|edition|ed\.?|vol\.?)\s*[）)]?',
        r'[（(]?\s*(library\s*edition|special\s*edition)\s*[）)]?',
    ]
    
    for pattern in edition_patterns:
        match = re.search(pattern, normalized_title, re.IGNORECASE)
        if match:
            edition = match.group(1).strip()
            # 正規化（大文字小文字を統一、余分な空白を削除）
            edition = normalize_string(edition, preserve_spaces=False).lower()
            return edition
    
    return None


def extract_important_keywords(title: str) -> set:
    """
    タイトルから重要なキーワードを抽出する
    
    技術用語、専門用語、固有名詞など、本の内容を特徴づける重要な単語を抽出
    
    Args:
        title: タイトル文字列
        
    Returns:
        重要なキーワードのセット
    """
    if not title:
        return set()
    
    # タイトルを正規化（空白も含めて正規化）
    normalized_title = normalize_string(title, preserve_spaces=True)
    
    # さらに空白の正規化を徹底（全角空白も含む）
    # 全角空白（\u3000）を半角空白に統一
    normalized_title = re.sub(r'[\u3000\t\n\r]+', ' ', normalized_title)
    # 連続する空白を1つに
    normalized_title = re.sub(r' +', ' ', normalized_title).strip()
    
    # ストップワード（除去すべき単語）
    stop_words = {
        'の', 'に', 'は', 'を', 'と', 'が', 'から', 'より', 'による', 'について',
        'における', 'に関する', 'による', 'によるもの', 'についての',
        '入門', '基礎', '実践', '応用', '概論', '概説', '入門編', '基礎編',
        'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        'basics', 'introduction', 'guide', 'handbook'
    }
    
    # 版情報や巻情報を除去
    edition_keywords = {
        '文庫', '新書', '電子', '改訂', '増補', '新版', '普及', '限定', '特装',
        'hardcover', 'paperback', 'ebook', 'edition', 'vol', 'volume',
        '上', '中', '下', '上巻', '中巻', '下巻', '第', '巻', '冊'
    }
    
    # 省略記号「...」を特別に処理（空白に置換して部分一致を許容）
    normalized_title = re.sub(r'\.{2,}', ' ', normalized_title)
    
    # 記号を空白に置換
    cleaned = re.sub(r'[「」『』（）\(\)\[\]\{\}【】〈〉《》""''""''、。，．・：；！？]', ' ', normalized_title)
    
    # 空白で分割（空白の正規化を徹底）
    # 全角空白、半角空白、タブ、改行などすべてを半角空白に統一
    # 連続する空白を1つに統一してから分割
    cleaned = re.sub(r'[\s\u3000]+', ' ', cleaned).strip()
    words = cleaned.split() if cleaned else []
    
    # 重要なキーワードを抽出
    # 2文字以上で、ストップワードや版情報でない単語
    # 空白のみの単語を除外
    important_keywords = {
        w.strip() for w in words
        if w.strip()  # 空白のみの単語を除外
        and len(w.strip()) > 1
        and w.strip() not in stop_words
        and w.strip() not in edition_keywords
        and not w.strip().isdigit()  # 数字は除外
        and not re.match(r'^[A-Za-z]$', w.strip())  # 1文字の英字は除外
    }
    
    return important_keywords


def detect_important_keyword_difference(title1: str, title2: str) -> bool:
    """
    2つのタイトル間で重要なキーワードが異なるか検出
    
    「基礎編」と「応用編」のような違いを検出する
    
    Args:
        title1: 1つ目のタイトル
        title2: 2つ目のタイトル
        
    Returns:
        重要なキーワードが異なる場合はTrue、そうでない場合はFalse
    """
    keywords1 = extract_important_keywords(title1)
    keywords2 = extract_important_keywords(title2)
    
    # どちらかが空の場合は判定できない（空白の違いなどは許容）
    if not keywords1 or not keywords2:
        return False
    
    # 共通のキーワード
    common = keywords1.intersection(keywords2)
    
    # 片方にしかない重要なキーワード
    only1 = keywords1 - keywords2
    only2 = keywords2 - keywords1
    
    # 共通のキーワード
    common = keywords1.intersection(keywords2)
    
    # 片方にしかない重要なキーワード
    only1 = keywords1 - keywords2
    only2 = keywords2 - keywords1
    
    # 空白の違いによる誤検出を防ぐ
    # キーワードが完全に同じ場合は不一致としない
    if keywords1 == keywords2:
        return False
    
    # 省略記号による部分的な欠損を許容
    # 片方のキーワードがもう片方のサブセットの場合は許容
    if keywords1.issubset(keywords2) or keywords2.issubset(keywords1):
        # サブセット関係の場合は、共通キーワードが十分にあると判断
        # より寛容な閾値（50%）を設定
        if len(common) >= min(len(keywords1), len(keywords2)) * 0.5:
            return False
    
    # 省略記号による部分的な欠損をさらに許容
    # 片方のキーワードがもう片方の大部分を含む場合も許容
    if len(common) >= max(len(keywords1), len(keywords2)) * 0.6:
        # 共通キーワードが最大セットの60%以上ある場合は許容
        return False
    
    # 共通のキーワード
    common = keywords1.intersection(keywords2)
    
    # 片方にしかない重要なキーワード
    only1 = keywords1 - keywords2
    only2 = keywords2 - keywords1
    
    # 重要なキーワードが異なるかどうかを判定
    # 共通キーワードが少なく、片方にしかないキーワードが多い場合は異なる
    if len(common) == 0 and (len(only1) > 0 or len(only2) > 0):
        return True
    
    # 共通キーワードより、片方にしかないキーワードの方が多い場合は異なる
    if len(only1) + len(only2) > len(common) * 1.5:
        return True
    
    # 「基礎編」と「応用編」のような違いを検出
    # 「編」を含む単語が異なる場合は異なる本と判定
    title1_norm = normalize_string(title1, preserve_spaces=True)
    title2_norm = normalize_string(title2, preserve_spaces=True)
    
    # 「編」を含む単語を抽出
    pattern = r'(\S+編)'
    matches1 = set(re.findall(pattern, title1_norm))
    matches2 = set(re.findall(pattern, title2_norm))
    
    if matches1 and matches2 and matches1 != matches2:
        return True
    
    # 同様に「版」「巻」などもチェック（ただし版は既に別途チェック済み）
    # 「上」「中」「下」などの違いもチェック
    volume_words = {'上', '中', '下', '上巻', '中巻', '下巻'}
    vol1 = set(w for w in keywords1 if w in volume_words)
    vol2 = set(w for w in keywords2 if w in volume_words)
    
    if vol1 and vol2 and vol1 != vol2:
        return True
    
    return False


def compute_name_similarity(author1, author2):
    """
    著者名の類似度を計算
    
    正規化後に類似度を計算する
    """
    if not author1 or not author2:
        return 0.0
    
    # 著者名を正規化
    author1_norm = normalize_author_name(author1)
    author2_norm = normalize_author_name(author2)
    
    if not author1_norm or not author2_norm:
        return 0.0
    
    return jaro_winkler_similarity(author1_norm, author2_norm)


def compute_keyword_overlap(title1, title2):
    """
    タイトル間のキーワード重複率を計算
    
    正規化後にキーワードを抽出して比較
    """
    if not title1 or not title2:
        return 0.0
    
    # タイトルを正規化
    title1_norm = normalize_string(title1, preserve_spaces=True)
    title2_norm = normalize_string(title2, preserve_spaces=True)
    
    # ストップワード（より包括的なリスト）
    stop_words = {
        'の', 'に', 'は', 'を', 'と', 'が', 'から', 'より', 'による', 'について',
        'における', 'に関する', 'による', 'によるもの', 'についての',
        'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'
    }
    
    def extract_keywords(title):
        """キーワードを抽出する"""
        # 記号を空白に置換（より包括的な記号リスト）
        # 括弧類、引用符、句読点など
        cleaned = re.sub(r'[「」『』（）\(\)\[\]\{\}【】〈〉《》""''""''、。，．・：；！？]', ' ', title)
        
        # 空白で分割
        words = cleaned.split()
        
        # ストップワードを除去し、1文字以上の単語のみを抽出
        keywords = {w for w in words if w not in stop_words and len(w) > 1}
        
        return keywords
    
    keywords1 = extract_keywords(title1_norm)
    keywords2 = extract_keywords(title2_norm)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Jaccard係数
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0


def normalize_pubdate(pubdate_str: str) -> str:
    """
    出版日を正規化する
    
    「1998.07」と「1998.7」のような違いを統一
    """
    if not pubdate_str:
        return ""
    
    # 基本的な正規化
    pubdate = normalize_string(pubdate_str, preserve_spaces=False)
    
    # 日付の正規化パターン
    # YYYY.MM または YYYY.M を YYYY.MM に統一
    date_pattern = r'(\d{4})\.(\d{1,2})'
    match = re.search(date_pattern, pubdate)
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2)  # 1桁の月を2桁に
        return f"{year}.{month}"
    
    # YYYY-MM 形式
    date_pattern2 = r'(\d{4})-(\d{1,2})'
    match = re.search(date_pattern2, pubdate)
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2)
        return f"{year}.{month}"
    
    # YYYY年MM月 形式
    date_pattern3 = r'(\d{4})年(\d{1,2})月'
    match = re.search(date_pattern3, pubdate)
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2)
        return f"{year}.{month}"
    
    return pubdate


def extract_features(record1, record2):
    """2つのレコードから特徴量を抽出"""
    features = {}
    
    # タイトルを正規化してから類似度を計算
    title1_norm = normalize_string(record1.get('title', ''), preserve_spaces=True)
    title2_norm = normalize_string(record2.get('title', ''), preserve_spaces=True)
    
    # 1. 基本的な文字列類似度（正規化後）
    features['title_similarity'] = jaro_winkler_similarity(title1_norm, title2_norm)
    
    # 2. 出版情報の一致
    features['same_publisher'] = int(
        normalize_publisher(record1.get('publisher', '')) == 
        normalize_publisher(record2.get('publisher', '')))
    
    # 出版日を正規化してから比較
    pubdate1_norm = normalize_pubdate(record1.get('pubdate', ''))
    pubdate2_norm = normalize_pubdate(record2.get('pubdate', ''))
    features['same_pubdate'] = int(pubdate1_norm == pubdate2_norm)
    
    # 3. 著者類似度
    features['author_similarity'] = compute_name_similarity(
        record1.get('author', ''), record2.get('author', ''))
    
    # 4. シリーズ情報（正規化後のタイトルから抽出）
    series1, vol1 = extract_series_info(title1_norm)
    series2, vol2 = extract_series_info(title2_norm)
    
    # シリーズ名の比較も正規化してから
    if series1 and series2:
        series1_norm = normalize_string(series1, preserve_spaces=True)
        series2_norm = normalize_string(series2, preserve_spaces=True)
        features['same_series'] = int(series1_norm == series2_norm)
    else:
        features['same_series'] = 0
    
    features['same_volume'] = int(vol1 == vol2 and vol1 is not None)
    
    # 5. キーワード重複（正規化後のタイトルを使用）
    features['keyword_overlap'] = compute_keyword_overlap(title1_norm, title2_norm)
        
    # 6. タイトルの長さの差（正規化後のタイトルで計算）
    len1 = len(title1_norm)
    len2 = len(title2_norm)
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
        author1_norm = normalize_author_name(record1.get('author', ''))
        author2_norm = normalize_author_name(record2.get('author', ''))
        pub1_norm = normalize_publisher(record1.get('publisher', ''))
        pub2_norm = normalize_publisher(record2.get('publisher', ''))
        date1_norm = normalize_pubdate(record1.get('pubdate', ''))
        date2_norm = normalize_pubdate(record2.get('pubdate', ''))
        
        if (author1_norm == author2_norm and 
            pub1_norm == pub2_norm and
            date1_norm == date2_norm and
            all([author1_norm, pub1_norm, date1_norm]) and
            all([author2_norm, pub2_norm, date2_norm])):
            return 1, 0.95
        return 0, 0.1
    
    # タイトルを正規化してから類似度を計算
    title1_norm = normalize_string(title1, preserve_spaces=True)
    title2_norm = normalize_string(title2, preserve_spaces=True)
    
    similarity = jaro_winkler_similarity(title1_norm, title2_norm)
    prediction = 1 if similarity >= threshold else 0
    confidence = similarity if prediction == 1 else 1 - similarity
    
    return prediction, confidence


def hybrid_prediction(record1, record2, ml_model, feature_names, threshold=0.85):
    """
    機械学習と文字列類似度を状況に応じて使い分ける予測関数
    
    不一致を強制するルール:
    1. 版が異なる場合
    2. シリーズ名が同じで巻数が異なる場合
    3. 重要なキーワードが異なる場合
    """
    
    # データの欠損状況を確認
    missing_data = any([
        not record1.get('author'), not record2.get('author'),
        not record1.get('publisher'), not record2.get('publisher'),
        not record1.get('pubdate'), not record2.get('pubdate')
    ])
    
    # タイトルの有無を確認
    title_present = bool(record1.get('title')) and bool(record2.get('title'))
    
    # タイトルを正規化
    title1_norm = ""
    title2_norm = ""
    if title_present:
        title1_norm = normalize_string(record1.get('title', ''), preserve_spaces=True)
        title2_norm = normalize_string(record2.get('title', ''), preserve_spaces=True)
    
    # ========================================================================
    # 不一致を強制するルール（機械学習の結果に関わらず）
    # ========================================================================
    
    if title_present:
        # ルール1: 版が異なる場合は不一致
        edition1 = extract_edition_info(title1_norm)
        edition2 = extract_edition_info(title2_norm)
        
        if edition1 and edition2 and edition1 != edition2:
            # 版が異なる場合は不一致
            return 0, 0.1
        
        # 版が片方にしかない場合も注意が必要だが、ここでは判定を保留
        # （片方だけに版情報がある場合は、同じ本の可能性もあるため）
    
    # ルール2: シリーズ名が同じで巻数が異なる場合は不一致
    if title_present:
        series1, vol1 = extract_series_info(title1_norm)
        series2, vol2 = extract_series_info(title2_norm)
        
        if series1 and series2:
            series1_norm = normalize_string(series1, preserve_spaces=True)
            series2_norm = normalize_string(series2, preserve_spaces=True)
            
            # シリーズ名が同じ
            if series1_norm == series2_norm:
                # 巻数が両方存在し、異なる場合は不一致
                if vol1 is not None and vol2 is not None and vol1 != vol2:
                    return 0, 0.1
                # 片方に巻数があり、もう片方にない場合も不一致の可能性が高い
                if (vol1 is not None and vol2 is None) or (vol1 is None and vol2 is not None):
                    # ただし、巻数表記の形式が異なる可能性もあるため、確信度は低めに
                    return 0, 0.3
    
    # ルール3: タイトルが片方のみ欠損している場合は不一致
    title1_empty = not bool(record1.get('title', '').strip())
    title2_empty = not bool(record2.get('title', '').strip())
    
    if title1_empty != title2_empty:  # 片方のみ欠損
        return 0, 0.1
    
    # ルール4: 重要なキーワードが異なる場合は不一致
    # ただし、両方のタイトルが存在する場合のみ
    # また、タイトルの類似度が高い場合は許容（空白の違いなど）
    if title_present and not title1_empty and not title2_empty:
        title_sim = jaro_winkler_similarity(title1_norm, title2_norm)
        # タイトルの類似度が非常に高い場合は、キーワードの違いを許容
        if title_sim < 0.95:  # 類似度が95%未満の場合のみキーワードチェック
            if detect_important_keyword_difference(title1_norm, title2_norm):
                return 0, 0.2
    
    # ========================================================================
    # 通常の判定ロジック
    # ========================================================================
    
    # 文字列類似度の計算
    title_similarity = 0
    if title_present:
        title_similarity = jaro_winkler_similarity(title1_norm, title2_norm)
    
    # 特別なケース：タイトル以外の情報が非常に類似（シリーズものなど）
    author_match = compute_name_similarity(record1.get('author', ''), record2.get('author', '')) > 0.9
    publisher_match = normalize_publisher(record1.get('publisher', '')) == normalize_publisher(record2.get('publisher', ''))
    date_match = normalize_pubdate(record1.get('pubdate', '')) == normalize_pubdate(record2.get('pubdate', ''))
    
    # 判断ロジック
    if missing_data or not title_present:
        # データ不足の場合はシンプルな文字列類似度または他フィールドの一致で判断
        return string_similarity_prediction(record1, record2, threshold)
    
    # 十分なデータがある場合は機械学習モデルを使用
    features_dict = extract_features(record1, record2)
    features_values = [features_dict[name] for name in feature_names]
    
    ml_prediction = ml_model.predict([features_values])[0]
    ml_confidence = ml_model.predict_proba([features_values])[0][1]
    
    # 特定条件では機械学習の結果を上書き（一致を強制するルール）
    if ml_prediction == 0:
        # 機械学習が「不一致」と判断した場合の特別ルール
        
        # ルール1: タイトルは異なるが他の全情報が一致（シリーズものの可能性）
        # ただし、巻数が異なる場合は除外（上で既にチェック済み）
        if author_match and publisher_match and date_match:
            # シリーズ情報を確認
            series1, vol1 = extract_series_info(title1_norm)
            series2, vol2 = extract_series_info(title2_norm)
            if series1 and series2:
                series1_norm = normalize_string(series1, preserve_spaces=True)
                series2_norm = normalize_string(series2, preserve_spaces=True)
                # シリーズ名が同じで、巻数も同じ（または両方なし）の場合のみ一致
                if series1_norm == series2_norm:
                    if (vol1 == vol2) or (vol1 is None and vol2 is None):
                        return 1, 0.9
                
        # ルール2: タイトルの類似度が非常に高い場合
        if title_similarity > 0.9:
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
