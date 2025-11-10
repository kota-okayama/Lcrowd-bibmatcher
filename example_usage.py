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
    
    result_custom = match_records(record1, record2, threshold=0.9, measure_time=True)
    print(f"閾値0.9での判定結果: {'一致' if result_custom['label'] == 1 else '不一致'}")
    print(f"確信度: {result_custom['confidence']:.4f}")
    print()



if __name__ == "__main__":
    # 各例を実行
    example_single_match()

