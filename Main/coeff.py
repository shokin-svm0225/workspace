import pandas as pd

# CSVファイルの読み込み
file_path = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/相関係数/データ/0/questionnaire_fusion_missing_0_P.csv'  # アップロードされたファイルのパス
data = pd.read_csv(file_path)

# '病名'カラムを除外
correlation_data = data.drop(columns=['病名'])

# 相関係数行列を計算
correlation_matrix = correlation_data.corr()

# 結果をCSVファイルとして保存
output_file_path = 'correlation_matrix_0_P.csv'  # 保存先ファイル名
correlation_matrix.to_csv(output_file_path)

print(f"相関係数行列を {output_file_path} に保存しました。")