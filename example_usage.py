"""
書誌レコード一致判定関数の使用例
"""
from match_api import match_records, match_records_batch

# 例1: 単一ペアの判定
def example_single_match():
    """単一ペアの判定例"""
    print("=== 例1: 単一ペアの判定 ===")
    
    record1 = {
        "title": "絵画における真理 下 叢書・ウニベルシタス 591",
        "author": "ジャック・デリダ/〔著〕",
        "publisher": "東京:法政大学出版局",
        "pubdate": "1998.07"
    }
    
    record2 = {
        "title": "絵画における真理 下  [叢書・ウニ] (591)",
        "author": "ジャック・デリダ/[著]",
        "publisher": " 法政大学出版局",    
        "pubdate": "1998.7"
    }
    
    # 処理時間を計測しながら判定
    result = match_records(record1, record2, measure_time=True)
    
    print(f"判定結果: {'一致' if result['label'] == 1 else '不一致'}")
    print(f"確信度: {result['confidence']:.4f}")
    print(f"使用手法: {result['method']}")
    print(f"処理時間: {result.get('processing_time_ms', 0):.2f}ms")
    print()


# 例2: バッチ処理
def example_batch_match():
    """複数ペアの一括判定例"""
    print("=== 例2: 複数ペアの一括判定 ===")
    
    record_pairs = [
        [
            {
                "title": "ハンニバル戦争",
                "author": "佐藤 賢一/著",
                "publisher": "東京:中央公論新社",
                "pubdate": "2016.01"
            },
            {
                "title": "ハンニバル戦争",
                "author": "佐藤 賢一",
                "publisher": "中央公論新社",
                "pubdate": "2016.01"
            },
            "example-id-1"
        ],
        [
            {
                "title": "こころ 夏目漱石",
                "author": "姜 尚中/著",
                "publisher": "東京:NHK出版",
                "pubdate": "2014.05"
            },
            {
                "title": "連写",
                "author": "今野 敏/著",
                "publisher": "朝日新聞出版",
                "pubdate": "2014.2"
            },
            "example-id-2"
        ]
    ]
    
    # バッチ処理（処理時間を計測）
    result = match_records_batch(record_pairs, measure_time=True)
    
    print(f"総件数: {result['total_count']}")
    print(f"総処理時間: {result.get('total_processing_time_ms', 0):.2f}ms")
    print(f"平均処理時間: {result.get('average_processing_time_ms', 0):.2f}ms")
    print()
    
    for i, r in enumerate(result['results']):
        print(f"ペア {i+1} (ID: {r.get('id', 'N/A')}):")
        print(f"  判定: {'一致' if r['label'] == 1 else '不一致'}")
        print(f"  確信度: {r['confidence']:.4f}")
        print(f"  手法: {r['method']}")
        print()


# 例3: カスタムモデルファイルの使用
def example_custom_model():
    """カスタムモデルファイルの使用例"""
    print("=== 例3: カスタムモデルファイルの使用 ===")
    
    record1 = {
        "title": "テストタイトル",
        "author": "テスト著者",
        "publisher": "テスト出版社",
        "pubdate": "2024.01"
    }
    
    record2 = {
        "title": "テストタイトル",
        "author": "テスト著者",
        "publisher": "テスト出版社",
        "pubdate": "2024.01"
    }
    
    # カスタムモデルファイルパスを指定
    # model_file = "/path/to/your/model.pkl"
    # result = match_records(record1, record2, model_file=model_file)
    
    # デフォルトモデルを使用
    result = match_records(record1, record2)
    
    print(f"判定結果: {'一致' if result['label'] == 1 else '不一致'}")
    print()


if __name__ == "__main__":
    # 各例を実行
    example_single_match()
    example_batch_match()
    example_custom_model()


