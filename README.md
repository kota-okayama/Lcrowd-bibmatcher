# 書誌レコード判定システム

## 1. 概要

このシステムは、2つの書誌レコード（タイトル、著者名、出版社名、出版日など）が同じ本を指しているかを判定するPythonライブラリです。

### 判定手法

1. **機械学習モデル**: RandomForestClassifierによる判定
2. **ルールベース**: 版情報、巻数、重要なキーワードの違いを検出
3. **文字列類似度**: Jaro-Winkler類似度による判定

### 判定の流れ

```
入力: 2つの書誌レコード
  ↓
1. 不一致を強制するルールチェック
   - 版が異なる場合
   - シリーズ名が同じで巻数が異なる場合
   - 重要なキーワードが異なる場合
  ↓
2. データの欠損状況を確認
  ↓
3. 十分なデータがある場合
   → 機械学習モデルで判定
   → 特定条件で結果を上書き
  ↓
4. データが不足している場合
   → 文字列類似度で判定
  ↓
出力: 判定結果（一致/不一致、確信度、使用手法）
```

## 2. 使用方法

### 基本的な使用例

```python
from match_api import match_records

# 2つの書誌レコードを準備
record1 = {
    "title": "絵画における真理 下 叢書・ウニベルシタス 591",
    "author": "ジャック・デリダ/〔著〕",
    "publisher": "東京:法政大学出版局",
    "pubdate": "1998.07"
}

record2 = {
    "title": "絵画における真理 下  [叢書・ウニ] (591)",
    "author": "ジャック・デリダ/[著]",
    "publisher": "法政大学出版局",
    "pubdate": "1998.7"
}

# 判定を実行
result = match_records(record1, record2)

# 結果を確認
print(f"判定: {'一致' if result['label'] == 1 else '不一致'}")
print(f"確信度: {result['confidence']:.4f}")
print(f"使用手法: {result['method']}")
```


## 3. APIリファレンス

### `match_records()`

2つの書誌レコードが一致するか判定する関数です。

#### パラメータ

- **record1** (`Dict[str, str]`): 1つ目のレコード
  - 必須フィールド: `title`, `author`, `publisher`, `pubdate`
  - フィールドは空文字列でも可（欠損データとして処理）
  
- **record2** (`Dict[str, str]`): 2つ目のレコード
  - 必須フィールド: `title`, `author`, `publisher`, `pubdate`
  - フィールドは空文字列でも可（欠損データとして処理）

- **model_file** (`Optional[str]`): モデルファイルのパス
  - デフォルト: `bibliography_matcher.pkl`（同じディレクトリ内）
  - カスタムモデルを使用する場合に指定

- **threshold** (`float`): 文字列類似度の閾値
  - デフォルト: `0.85`
  - 範囲: `0.0` ～ `1.0`
  - 高いほど厳格な判定（一致と判定されにくい）

- **measure_time** (`bool`): 処理時間を計測するか
  - デフォルト: `False`
  - `True`にすると結果に`processing_time_ms`が含まれる

#### 戻り値

```python
{
    "label": int,           # 0=不一致, 1=一致
    "confidence": float,     # 確信度 (0.0-1.0)
    "method": str,          # 使用した手法
                            # - "machine_learning": 機械学習モデル
                            # - "hybrid_rules": ルールベース
                            # - "string_similarity": 文字列類似度
    "ml_prediction": int,   # 機械学習の判定結果
    "ml_confidence": float, # 機械学習の確信度
    "str_prediction": int,  # 文字列類似度の判定結果
    "str_confidence": float,# 文字列類似度の確信度
    "processing_time_ms": float  # 処理時間（ミリ秒、measure_time=Trueの場合）
}
```

#### 使用例

```python
# 基本的な使用
result = match_records(record1, record2)

# 閾値を調整
result = match_records(record1, record2, threshold=0.9)

# 処理時間を計測
result = match_records(record1, record2, measure_time=True)

# カスタムモデルを使用
result = match_records(record1, record2, model_file="custom_model.pkl")
```


## 4. パラメータの調整

### 4.1 主要パラメータ一覧

| パラメータ | デフォルト値 | 変更箇所 | 影響 |
|-----------|------------|---------|------|
| `threshold` | 0.85 | `match_records()` 関数 | 文字列類似度の閾値 |
| 機械学習の閾値 | 0.5 | `hybrid_prediction()` 686行目（変更可能） | 機械学習モデルの判定閾値（確信度0.5以上で一致）|
| キーワードサブセット閾値 | 0.5 | `detect_important_keyword_difference()` 336行目 | 部分的な欠損の許容度 |
| キーワード共通閾値 | 0.6 | `detect_important_keyword_difference()` 341行目 | 共通キーワードの許容度 |
| タイトル類似度閾値（キーワードチェック） | 0.95 | `hybrid_prediction()` 659行目 | キーワードチェックのスキップ条件 |
| タイトル類似度閾値（一致強制ルール） | 0.9 | `hybrid_prediction()` 708行目 | 機械学習が不一致でも一致を強制する条件 |

**注意**: 
- **機械学習の閾値（0.5）**: これはscikit-learnのRandomForestClassifierの`predict()`メソッドのデフォルト閾値です。`predict_proba()`で確信度を取得し、カスタム閾値で判定することも可能です（詳細は「6.8 機械学習モデルの閾値の変更」を参照）。
- **タイトル類似度閾値（一致強制ルール）**: これは機械学習の閾値ではなく、機械学習が「不一致」と判定した場合でも、タイトルの類似度がこの値以上であれば「一致」を強制するルールベースの閾値です。

## 5. 判定ロジックの説明

### ハイブリッドアプローチ

このシステムは、機械学習モデルとルールベースの組み合わせ（ハイブリッドアプローチ）を使用しています。

### 判定の優先順位

#### 1. 不一致を強制するルール（最優先）

以下の条件に該当する場合、機械学習の結果に関わらず**不一致**と判定されます：

##### ルール1: 版が異なる場合
- 例: 「文庫版」と「新書版」、「電子版」と「通常版」
- 検出される版情報: 文庫版、新書版、電子版、改訂版、ハードカバー、ペーパーバックなど
- 実装箇所: `extract_edition_info()` 関数（187-221行目）

##### ルール2: シリーズ名が同じで巻数が異なる場合
- 例: 「Python入門シリーズ 第1巻」と「Python入門シリーズ 第2巻」
- 検出される巻数表記: 第X巻、Vol.X、上/中/下など
- 実装箇所: `extract_series_info()` 関数（120-184行目）、`hybrid_prediction()` 関数（627-644行目）

##### ルール3: タイトルが片方のみ欠損している場合
- 例: タイトルあり vs タイトルなし
- 実装箇所: `hybrid_prediction()` 関数（646-651行目）

##### ルール4: 重要なキーワードが異なる場合
- 例: 「機械学習入門」と「深層学習入門」
- 検出方法: タイトルから重要なキーワードを抽出し、違いを検出
- 実装箇所: `extract_important_keywords()` 関数（224-289行目）、`detect_important_keyword_difference()` 関数（292-383行目）
- **注意**: タイトルの類似度が95%以上の場合はこのチェックをスキップ（空白の違いなどを許容）

#### 2. データの欠損状況による分岐

##### データが不足している場合
- タイトルがない、または著者名・出版社名・出版日のいずれかが欠損
- → `string_similarity_prediction()` 関数を使用（551-581行目）
- 文字列類似度または他のフィールドの一致で判定

##### 十分なデータがある場合
- → 機械学習モデルを使用
- → 特定条件で結果を上書き

#### 3. 一致を強制するルール

機械学習が「不一致」と判定した場合でも、以下の条件に該当すれば**一致**と判定されます：

##### ルール1: シリーズ名が同じで巻数も同じ
- タイトルは異なるが、著者名・出版社名・出版日が一致
- シリーズ名が同じで、巻数も同じ（または両方なし）

##### ルール2: タイトルの類似度が非常に高い
- タイトルの類似度が90%以上

### 文字列正規化

すべての文字列比較は、正規化後に実行されます：

- **Unicode正規化**: NFKC形式（全角英数字を半角に変換）
- **空白の統一**: 全角空白、タブ、改行を半角空白に統一
- **記号の処理**: 括弧、引用符などの記号を適切に処理

実装箇所: `normalize_string()` 関数（22-55行目）

## 6. 結果の調整方法

判定結果を調整したい場合、以下の箇所を変更できます。

### 6.1 文字列類似度の閾値

**変更箇所**: `match_records()` 関数の `threshold` パラメータ

```python
# より厳格な判定（一致と判定されにくい）
result = match_records(record1, record2, threshold=0.9)

# より緩い判定（一致と判定されやすい）
result = match_records(record1, record2, threshold=0.7)
```

**影響**: 
- 高い値: より厳格な判定（不一致が増える）
- 低い値: より緩い判定（一致が増える）

### 6.2 版情報の検出パターン

**変更箇所**: `extract_edition_info()` 関数（187-221行目）

```python
# 版情報のパターンを追加・変更
edition_patterns = [
    r'[（(]?\s*(文庫版|新書版|電子版|...)\s*[）)]?',
    # 新しいパターンを追加
    r'[（(]?\s*(翻訳版|日本語版)\s*[）)]?',  # 例: 翻訳版を追加
]
```

**影響**: 新しい版情報を検出できるようになる

### 6.3 重要なキーワードの判定閾値

**変更箇所**: `detect_important_keyword_difference()` 関数（292-383行目）

```python
# サブセット関係の許容閾値（336行目付近）
if len(common) >= min(len(keywords1), len(keywords2)) * 0.5:  # 0.5を変更
    return False

# 共通キーワードの許容閾値（341行目付近）
if len(common) >= max(len(keywords1), len(keywords2)) * 0.6:  # 0.6を変更
    return False
```

**影響**:
- 値を大きくする: より多くの違いを許容（一致が増える）
- 値を小さくする: より厳格な判定（不一致が増える）

### 6.4 タイトル類似度の閾値（キーワードチェックのスキップ）

**変更箇所**: `hybrid_prediction()` 関数（659行目付近）

```python
# タイトルの類似度が非常に高い場合は、キーワードの違いを許容
if title_sim < 0.95:  # 0.95を変更（例: 0.9にするとより厳格に）
    if detect_important_keyword_difference(title1_norm, title2_norm):
        return 0, 0.2
```

**影響**:
- 値を大きくする: より多くのケースでキーワードチェックをスキップ（一致が増える）
- 値を小さくする: より多くのケースでキーワードチェックを実行（不一致が増える）

### 6.5 一致を強制するルールの閾値

**変更箇所**: `hybrid_prediction()` 関数（708行目付近）

```python
# タイトルの類似度が非常に高い場合
if title_similarity > 0.9:  # 0.9を変更
    return 1, title_similarity
```

**影響**:
- 値を大きくする: より厳格な条件で一致を強制（一致が減る）
- 値を小さくする: より緩い条件で一致を強制（一致が増える）

**注意**: これは**機械学習の閾値ではありません**。機械学習が「不一致」と判定した場合でも、タイトルの類似度がこの値以上であれば「一致」を強制する**ルールベースの閾値**です。

### 6.8 機械学習モデルの閾値の変更

**デフォルト動作**: 機械学習モデル（RandomForestClassifier）の`predict()`メソッドは、内部で**0.5**を閾値として使用します（686行目）。

- 確信度が0.5以上 → 「一致」（label=1）
- 確信度が0.5未満 → 「不一致」（label=0）

**別の閾値で判定する方法**:

`predict()`の代わりに`predict_proba()`の結果を使用して、カスタム閾値で判定できます。

#### 方法1: `hybrid_prediction()`関数を修正（推奨）

**変更箇所**: `hybrid_prediction()` 関数（686-687行目付近）

```python
# 変更前
ml_prediction = ml_model.predict([features_values])[0]
ml_confidence = ml_model.predict_proba([features_values])[0][1]

# 変更後（カスタム閾値0.7を使用）
ml_confidence = ml_model.predict_proba([features_values])[0][1]
ml_custom_threshold = 0.7  # カスタム閾値（例: 0.7）
ml_prediction = 1 if ml_confidence >= ml_custom_threshold else 0
```

**影響**:
- 閾値を上げる（例: 0.5 → 0.7）: より厳格な判定（一致が減る）
- 閾値を下げる（例: 0.5 → 0.3）: より緩い判定（一致が増える）

#### 方法2: `match_records()`関数にパラメータを追加

**変更箇所**: `match_records()` 関数と `hybrid_prediction()` 関数

```python
# match_records()関数のシグネチャを変更
def match_records(
    record1: Dict[str, str],
    record2: Dict[str, str],
    model_file: Optional[str] = None,
    threshold: float = 0.85,
    ml_threshold: float = 0.5,  # 機械学習の閾値を追加
    measure_time: bool = False
) -> Dict[str, Any]:
    # ...
    # hybrid_prediction()にml_thresholdを渡す
    hybrid_pred, hybrid_conf = hybrid_prediction(
        record1, record2, model, feature_names, threshold, ml_threshold
    )
    # ...

# hybrid_prediction()関数のシグネチャを変更
def hybrid_prediction(
    record1, record2, ml_model, feature_names, 
    threshold=0.85, ml_threshold=0.5  # 機械学習の閾値を追加
):
    # ...
    ml_confidence = ml_model.predict_proba([features_values])[0][1]
    ml_prediction = 1 if ml_confidence >= ml_threshold else 0  # カスタム閾値で判定
    # ...
```

**使用例**:
```python
# より厳格な判定（確信度0.7以上で一致）
result = match_records(record1, record2, ml_threshold=0.7)

# より緩い判定（確信度0.3以上で一致）
result = match_records(record1, record2, ml_threshold=0.3)
```

**現在の確信度を確認する方法**:
```python
result = match_records(record1, record2)
print(f"機械学習の確信度: {result['ml_confidence']:.4f}")
print(f"機械学習の判定（0.5閾値）: {'一致' if result['ml_prediction'] == 1 else '不一致'}")

# カスタム閾値で判定
custom_threshold = 0.7
custom_prediction = 1 if result['ml_confidence'] >= custom_threshold else 0
print(f"カスタム判定（{custom_threshold}閾値）: {'一致' if custom_prediction == 1 else '不一致'}")
```

### 6.6 ストップワードの追加

**変更箇所**: `extract_important_keywords()` 関数（249-255行目）

```python
stop_words = {
    'の', 'に', 'は', 'を', 'と', 'が', ...,
    # 新しいストップワードを追加
    '概論', '入門',  # 例: より多くの汎用語を除外
}
```

**影響**: より多くの汎用語を除外し、重要なキーワードのみを抽出

### 6.7 版情報キーワードの追加

**変更箇所**: `extract_important_keywords()` 関数（258-262行目）

```python
edition_keywords = {
    '文庫', '新書', '電子', ...,
    # 新しい版情報を追加
    '翻訳', '日本語',  # 例: 翻訳版を版情報として扱う
}
```

**影響**: 新しい版情報を除外し、重要なキーワードの抽出に影響



## 7 調整の方針

### 一致を増やしたい場合

1. `threshold`を下げる（例: 0.85 → 0.7）
2. **機械学習の閾値を下げる**（例: 0.5 → 0.3）
3. キーワードサブセット閾値を上げる（例: 0.5 → 0.7）
4. キーワード共通閾値を上げる（例: 0.6 → 0.8）
5. タイトル類似度閾値（キーワードチェック）を上げる（例: 0.95 → 0.98）
6. タイトル類似度閾値（一致強制）を下げる（例: 0.9 → 0.85）

### 不一致を増やしたい場合（より厳格な判定）

1. `threshold`を上げる（例: 0.85 → 0.9）
2. **機械学習の閾値を上げる**（例: 0.5 → 0.7）
3. キーワードサブセット閾値を下げる（例: 0.5 → 0.3）
4. キーワード共通閾値を下げる（例: 0.6 → 0.4）
5. タイトル類似度閾値（キーワードチェック）を下げる（例: 0.95 → 0.9）
6. タイトル類似度閾値（一致強制）を上げる（例: 0.9 → 0.95）
