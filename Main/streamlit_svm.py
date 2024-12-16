#session_stateを使う場合
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import cv2
import csv
import datetime
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu

TEST_DATA_RATIO = 0.3
SAVE_TRAINED_DATA_PATH = "svm_data.xml"

with st.sidebar:
    selected = option_menu(
        menu_title="メニュー",  # サイドバータイトル
        options=['ホーム', '実験', '本日の発表', '発表履歴', 'データセット', 'プログラム', 'データ分析', '実験結果の保存', '質問表'],
        icons=["house", "play", "rocket", "rocket", "puzzle-fill", "chat-dots", "tag-fill", "chat-dots", "quora"],
        menu_icon="power",
        default_index=0,
    )

# 選択されたページに応じてコンテンツを表示
if selected == 'ホーム':
    st.title('ホーム')
    st.header("研究内容")
    st.write("AIモデルの一つであるSVMを用い、整形外科での診断補助システム開発を行う")
    st.header("背景")
    st.write("整形外科の患者が訴える典型的症状に、疼痛（慢性痛の総称）がある。人口の20％以上が何らかの疼痛を感じているとの報告もある程、普遍的な症状の一つ。  \n医師による疼痛原因の診断法は複数開発されている一方、手法により精度の差があることが知られている。")
    st.header("目的")
    st.write("人工知能による疼痛診断の自動化を目指す。  \n医学的診断を人工知能で行う際の制約として、「判断根拠を説明できなければいけない」、また「各医療機関が持つデータ量は限定的」というものがある。  \nこれらを解決するため、「少ないデータでも高い精度を達成しやすい」、そして「判断理由の解釈も比較的容易」という特徴を持つ\nSVMを用いることで、高精度かつ説明可能なAIによる診断補助システム構築を目指す。")
    st.header("質問項目の説明")
    st.write("PainDETECT")
    # CSVファイルのパスを指定
    csv_file_path_5 = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/painditect_質問.csv'  # ファイルパスを指定
    df_paindetect = pd.read_csv(csv_file_path_5)
    # データフレームを表示
    st.dataframe(df_paindetect)
    st.write("BS-POP")
    # CSVファイルのパスを指定
    csv_file_path_6 = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_質問.csv'  # ファイルパスを指定
    df_bspop = pd.read_csv(csv_file_path_6)
    # データフレームを表示
    st.dataframe(df_bspop)
    # CSVファイルのパスを指定
    csv_file_path_7 = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_痛みの経過.csv'  # ファイルパスを指定
    df_bspop_answer = pd.read_csv(csv_file_path_7)
    # データフレームを表示
    st.dataframe(df_bspop_answer)


    st.header("参考サイト")
    st.write("- [Streamlit_documentation](https://docs.streamlit.io/): streamlitのドキュメント参考")

    st.header("今後の予定")
    st.write("- 重みの設定")
    st.write("- カーネルのパラメータの変更・設定（色々なカーネルで試す）")
    st.write("- カーネルの関数設定")
    st.write("- 特徴量エンジニアリング")
    st.write("--- ランダムサーチ")
    st.write("- ハイパーパラメータ(C)のチューニング")
    st.write("- 遺伝的アルゴリズムを用いて、パラメータCを求める")
    st.write("--- クロスバリデーション（交差検証）")
    st.write("--- ランダムサーチ")
    st.write("--- グリッドサーチ")
    st.write("--- ベイズ最適化")
    st.write("- モデルの評価指標の見直し")
    st.write("- Scikit-Learnの学習")

    st.header("実験ログ")
    st.write("- 20241101 : streamlit上で実験ができるようにしました")
    st.write("- 20241102 : 今まで利用したCSVファイルのカラムを統一にしました")
    st.write("- 20241106 : 各質問項目における相関係数の出力をしました")
    st.write("- 20241107 : streamlit上で相関係数の出力・評価の確認を可能にしました")

elif selected == '実験':
    st.title('疼痛診断システムの開発')
    st.markdown('#### 侵害受容性疼痛')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_1")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df1 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df1)

    st.markdown('#### 神経障害性疼痛')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_2")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df2 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df2)

    st.markdown('#### 原因不明')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_3")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df3 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df3)

    #svmのプログラムを組み込む
    st.markdown('#### 使用するカラムの指定(painDETECT)')

    # 初期化
    if 'checkbox_states_1' not in st.session_state:
        st.session_state.checkbox_states_1 = {
            f"P{i}": False for i in range(1, 14)  # P1からP7まで初期化
        }

    # 全選択・全解除ボタン
    col_buttons = st.columns(2)
    if col_buttons[0].button('全選択', key='select_all_1'):
        for key in st.session_state.checkbox_states_1:
            st.session_state.checkbox_states_1[key] = True

    if col_buttons[1].button('全解除', key='deselect_all_1'):
        for key in st.session_state.checkbox_states_1:
            st.session_state.checkbox_states_1[key] = False

    # チェックボックスの表示（元のスタイルを維持）
    col_1 = st.columns(7)
    painDITECT_1 = col_1[0].checkbox(label='P1', value=st.session_state.checkbox_states_1["P1"], key="P1")
    painDITECT_2 = col_1[1].checkbox(label='P2', value=st.session_state.checkbox_states_1["P2"], key="P2")
    painDITECT_3 = col_1[2].checkbox(label='P3', value=st.session_state.checkbox_states_1["P3"], key="P3")
    painDITECT_4 = col_1[3].checkbox(label='P4', value=st.session_state.checkbox_states_1["P4"], key="P4")
    painDITECT_5 = col_1[4].checkbox(label='P5', value=st.session_state.checkbox_states_1["P5"], key="P5")
    painDITECT_6 = col_1[5].checkbox(label='P6', value=st.session_state.checkbox_states_1["P6"], key="P6")
    painDITECT_7 = col_1[6].checkbox(label='P7', value=st.session_state.checkbox_states_1["P7"], key="P7")

    col_2 = st.columns(6)
    painDITECT_8 = col_2[0].checkbox(label='P8', value=st.session_state.checkbox_states_1["P8"], key="P8")
    painDITECT_9 = col_2[1].checkbox(label='P9', value=st.session_state.checkbox_states_1["P9"], key="P9")
    painDITECT_10 = col_2[2].checkbox(label='P10', value=st.session_state.checkbox_states_1["P10"], key="P10")
    painDITECT_11 = col_2[3].checkbox(label='P11', value=st.session_state.checkbox_states_1["P11"], key="P11")
    painDITECT_12 = col_2[4].checkbox(label='P12', value=st.session_state.checkbox_states_1["P12"], key="P12")
    painDITECT_13 = col_2[5].checkbox(label='P13', value=st.session_state.checkbox_states_1["P13"], key="P13")

    # 状態を反映
    st.session_state.checkbox_states_1["P1"] = painDITECT_1
    st.session_state.checkbox_states_1["P2"] = painDITECT_2
    st.session_state.checkbox_states_1["P3"] = painDITECT_3
    st.session_state.checkbox_states_1["P4"] = painDITECT_4
    st.session_state.checkbox_states_1["P5"] = painDITECT_5
    st.session_state.checkbox_states_1["P6"] = painDITECT_6
    st.session_state.checkbox_states_1["P7"] = painDITECT_7
    st.session_state.checkbox_states_1["P8"] = painDITECT_8
    st.session_state.checkbox_states_1["P9"] = painDITECT_9
    st.session_state.checkbox_states_1["P10"] = painDITECT_10
    st.session_state.checkbox_states_1["P11"] = painDITECT_11
    st.session_state.checkbox_states_1["P12"] = painDITECT_12
    st.session_state.checkbox_states_1["P13"] = painDITECT_13


    st.markdown('#### 使用するカラムの指定(BSPOP)')

    # 初期化
    if 'checkbox_states_2' not in st.session_state:
        st.session_state.checkbox_states_2 = {
            f"D{i}": False for i in range(1, 19)  # D1からP19まで初期化
        }

    # 全選択・全解除ボタン
    col_buttons = st.columns(2)
    if col_buttons[0].button('全選択', key='select_all_2'):
        for key in st.session_state.checkbox_states_2:
            st.session_state.checkbox_states_2[key] = True

    if col_buttons[1].button('全解除', key='deselect_all_2'):
        for key in st.session_state.checkbox_states_2:
            st.session_state.checkbox_states_2[key] = False

    col_3 = st.columns(6)
    BSPOP_1 = col_3[0].checkbox(label='D1', value=st.session_state.checkbox_states_2["D1"], key="D1")
    BSPOP_2 = col_3[1].checkbox(label='D2', value=st.session_state.checkbox_states_2["D2"], key="D2")
    BSPOP_3 = col_3[2].checkbox(label='D3', value=st.session_state.checkbox_states_2["D3"], key="D3")
    BSPOP_4 = col_3[3].checkbox(label='D4', value=st.session_state.checkbox_states_2["D4"], key="D4")
    BSPOP_5 = col_3[4].checkbox(label='D5', value=st.session_state.checkbox_states_2["D5"], key="D5")
    BSPOP_6 = col_3[5].checkbox(label='D6', value=st.session_state.checkbox_states_2["D6"], key="D6")
    
    # 2行目のチェックボックス（D7〜D12）
    col_4 = st.columns(6)
    BSPOP_7 = col_4[0].checkbox(label='D7', value=st.session_state.checkbox_states_2["D7"], key="D7")
    BSPOP_8 = col_4[1].checkbox(label='D8', value=st.session_state.checkbox_states_2["D8"], key="D8")
    BSPOP_9 = col_4[2].checkbox(label='D9', value=st.session_state.checkbox_states_2["D9"], key="D9")
    BSPOP_10 = col_4[3].checkbox(label='D10', value=st.session_state.checkbox_states_2["D10"], key="D10")
    BSPOP_11 = col_4[4].checkbox(label='D11', value=st.session_state.checkbox_states_2["D11"], key="D11")
    BSPOP_12 = col_4[5].checkbox(label='D12', value=st.session_state.checkbox_states_2["D12"], key="D12")
    
    # 3行目のチェックボックス（D13〜D18）
    col_5 = st.columns(6)
    BSPOP_13 = col_5[0].checkbox(label='D13', value=st.session_state.checkbox_states_2["D13"], key="D13")
    BSPOP_14 = col_5[1].checkbox(label='D14', value=st.session_state.checkbox_states_2["D14"], key="D14")
    BSPOP_15 = col_5[2].checkbox(label='D15', value=st.session_state.checkbox_states_2["D15"], key="D15")
    BSPOP_16 = col_5[3].checkbox(label='D16', value=st.session_state.checkbox_states_2["D16"], key="D16")
    BSPOP_17 = col_5[4].checkbox(label='D17', value=st.session_state.checkbox_states_2["D17"], key="D17")
    BSPOP_18 = col_5[5].checkbox(label='D18', value=st.session_state.checkbox_states_2["D18"], key="D18")

    # 状態を反映
    st.session_state.checkbox_states_2["D1"] = BSPOP_1
    st.session_state.checkbox_states_2["D2"] = BSPOP_2
    st.session_state.checkbox_states_2["D3"] = BSPOP_3
    st.session_state.checkbox_states_2["D4"] = BSPOP_4
    st.session_state.checkbox_states_2["D5"] = BSPOP_5
    st.session_state.checkbox_states_2["D6"] = BSPOP_6
    st.session_state.checkbox_states_2["D7"] = BSPOP_7
    st.session_state.checkbox_states_2["D8"] = BSPOP_8
    st.session_state.checkbox_states_2["D9"] = BSPOP_9
    st.session_state.checkbox_states_2["D10"] = BSPOP_10
    st.session_state.checkbox_states_2["D11"] = BSPOP_11
    st.session_state.checkbox_states_2["D12"] = BSPOP_12
    st.session_state.checkbox_states_2["D13"] = BSPOP_13
    st.session_state.checkbox_states_2["D14"] = BSPOP_14
    st.session_state.checkbox_states_2["D15"] = BSPOP_15
    st.session_state.checkbox_states_2["D16"] = BSPOP_16
    st.session_state.checkbox_states_2["D17"] = BSPOP_17
    st.session_state.checkbox_states_2["D18"] = BSPOP_18

    st.markdown('#### 重みづけの指定')

    stocks = []
    if painDITECT_1:
        stocks.append('P1')
    if painDITECT_2:
        stocks.append('P2')
    if painDITECT_3:
        stocks.append('P3')
    if painDITECT_4:
        stocks.append('P4')
    if painDITECT_5:
        stocks.append('P5')
    if painDITECT_6:
        stocks.append('P6')
    if painDITECT_7:
        stocks.append('P7')
    if painDITECT_8:
        stocks.append('P8')
    if painDITECT_9:
        stocks.append('P9')
    if painDITECT_10:
        stocks.append('P10')
    if painDITECT_11:
        stocks.append('P11')
    if painDITECT_12:
        stocks.append('P12')
    if painDITECT_13:
        stocks.append('P13')
    if BSPOP_1:
        stocks.append('D1')
    if BSPOP_2:
        stocks.append('D2')
    if BSPOP_3:
        stocks.append('D3')
    if BSPOP_4:
        stocks.append('D4')
    if BSPOP_5:
        stocks.append('D5')
    if BSPOP_6:
        stocks.append('D6')
    if BSPOP_7:
        stocks.append('D7')
    if BSPOP_8:
        stocks.append('D8')
    if BSPOP_9:
        stocks.append('D9')
    if BSPOP_10:
        stocks.append('D10')
    if BSPOP_11:
        stocks.append('D11')
    if BSPOP_12:
        stocks.append('D12')
    if BSPOP_13:
        stocks.append('D13')
    if BSPOP_14:
        stocks.append('D14')
    if BSPOP_15:
        stocks.append('D15')
    if BSPOP_16:
        stocks.append('D16')
    if BSPOP_17:
        stocks.append('D17')
    if BSPOP_18:
        stocks.append('D18')

    weights = []
    
    # セッションステートの初期化
    if "weights" not in st.session_state:
        st.session_state.weights = {stock: 1.0 for stock in stocks}
    if "reset" not in st.session_state:
        st.session_state.reset = False

    # 重みの初期化
    if st.button("重みをリセット", key="weights_reset"):
        for stock in stocks:
            st.session_state.weights[stock] = 1.0  # 全ての重みを初期化
        st.session_state.reset = True

    # 動的にスライドバーを生成し、weightsに格納
    for column in stocks:
        if column not in st.session_state.weights:
            st.session_state.weights[column] = 1.0
        # セッションステートからスライダーの初期値を取得
        default_weight = st.session_state.weights[column]
        weight = st.slider(f"{column}の重み", min_value=-5.0, max_value=5.0, value=default_weight, step=0.1, key=f"slider_{column}")
        weights.append(weight)
        # スライダーの値をセッションステートに保存
        st.session_state.weights[column] = weight

    # データフレームを作成
    edited_df = pd.DataFrame({"columns": stocks, "weights": weights})

    # データフレームを表示
    st.markdown("### 重みづけデータフレーム")
    st.dataframe(edited_df)

    #データの加工方法の指定
    options = ['欠損値削除', '中央値補完', '平均値補完', 'k-NN法補完']

    # セレクトボックスを作成し、ユーザーの選択を取得
    data_processing = st.selectbox('欠損値補完の方法は？', options, index = None, placeholder="選択してください")

    if st.button("開始", help="実験の実行"):
        columns = edited_df["columns"].tolist()
        weights = edited_df["weights"].tolist()
        
        # データの分割
        df_nociceptive_train, df_nociceptive_test = train_test_split(
            df1[columns], test_size=TEST_DATA_RATIO, random_state=None
            )
        df_neuronociceptive_train, df_neuronociceptive_test = train_test_split(
            df2[columns], test_size=TEST_DATA_RATIO, random_state=None
            )
        df_unknown_train, df_unknown_test = train_test_split(
            df3[columns], test_size=TEST_DATA_RATIO, random_state=None
            )
    
        # 重みを適用して特徴量を調整
        df_nociceptive_train_weighted = df_nociceptive_train.mul(weights, axis=1)

        df_neuronociceptive_train_weighted = df_neuronociceptive_train.mul(weights, axis=1)

        df_unknown_train_weighted = df_unknown_train.mul(weights, axis=1)
        
    
        # トレーニングデータとラベルの作成
        datas = np.vstack(
            [
                df_nociceptive_train_weighted.values,
                df_neuronociceptive_train_weighted.values,
                df_unknown_train_weighted.values,
                ]
                ).astype(np.float32)
        
        labels1 = np.full(len(df_nociceptive_train_weighted), 1, np.int32)
        labels2 = np.full(len(df_neuronociceptive_train_weighted), 2, np.int32)
        labels3 = np.full(len(df_unknown_train_weighted), 3, np.int32)
        labels = np.concatenate([labels1, labels2, labels3]).astype(np.int32)
        
        # SVMモデルの作成とトレーニング
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setGamma(1)
        svm.setC(1)
        svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))
        svm.train(datas, cv2.ml.ROW_SAMPLE, labels)
        
        # モデルを保存
        svm.save(SAVE_TRAINED_DATA_PATH)
        
        test_datas = np.vstack(
            [
            df_nociceptive_test.values,
            df_neuronociceptive_test.values,
            df_unknown_test.values,
            ]
            ).astype(np.float32)
        
        test_labels1 = np.full(len(df_nociceptive_test), 1, np.int32)
        test_labels2 = np.full(len(df_neuronociceptive_test), 2, np.int32)
        test_labels3 = np.full(len(df_unknown_test), 3, np.int32)
        
        test_labels = np.concatenate([test_labels1, test_labels2, test_labels3]).astype(
            np.int32
            )
        
        # # データの標準化
        # scaler = StandardScaler()
        # datas = scaler.fit_transform(datas)
        # test_datas = scaler.transform(test_datas)
        
        # # 交差検証の実行
        # cross_val_scores = cross_val_score(svm, datas, labels, cv=5)
        # print("Cross-Validation Scores:", cross_val_scores)
        # print("Mean Cross-Validation Score:", cross_val_scores.mean())
        
        svm = cv2.ml.SVM_load(SAVE_TRAINED_DATA_PATH)
        _, predicted = svm.predict(test_datas)
        
        confusion_matrix = np.zeros((3, 3), dtype=int)
        
        for i in range(len(test_labels)):
            index1 = test_labels[i] - 1
            index2 = predicted[i][0] - 1
            confusion_matrix[int(index1)][int(index2)] += 1
            
        st.write("confusion matrix")
        st.table(confusion_matrix)

        score = np.sum(test_labels == predicted.flatten()) / len(test_labels)
            
        st.write("正答率:", score*100, "%")
            
        # 感度と特異度の計算
        sensitivity = np.zeros(3)
        specificity = np.zeros(3)
        
        for i in range(3):
            TP = confusion_matrix[i, i]
            FN = np.sum(confusion_matrix[i, :]) - TP
            FP = np.sum(confusion_matrix[:, i]) - TP
            TN = np.sum(confusion_matrix) - (TP + FN + FP)
            
            sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0
            
        # 感度と特異度の表示
        st.write("感度と特異度")
        st.write("（疼痛1:侵害受容性疼痛,疼痛2:神経障害性疼痛,疼痛3:不明）")
        for i in range(3):
            st.write(f"疼痛 {i+1}: 感度 = {sensitivity[i]:.4f}, 特異度 = {specificity[i]:.4f}")

        # 現在の日時を取得
        dt_now = datetime.datetime.now()

        # アップロードしたCSVファイルのパス
        LOG_FILE_PATH = '/Users/iwasho_0225/Desktop/workspace/pain_experiment/log/LOG_FILE.csv'

        # 新しいデータを1行にまとめる
        new_row = {
            'date': dt_now.strftime('%Y%m%d-%H%M%S'),
            'data_processing': data_processing,
            'use_columns': ', '.join(map(str, columns)),
            'weights': ', '.join(map(str, weights)),
            'score': str(score*100),
            'sensitivity': ', '.join(map(str, [sensitivity[0],sensitivity[1],sensitivity[2]])),
            'specificity': ', '.join(map(str, [specificity[0],specificity[1],specificity[2]]))
        }

        # CSVファイルに追記（既存のヘッダーを維持）
        with open(LOG_FILE_PATH, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=new_row.keys())

            # データを一行で追加
            writer.writerow(new_row)

elif selected == '本日の発表':
    st.title('発表内容')
    st.header("概要")
    st.write("- UIデザインの変更")
    st.write("--- 今までの発表内容(stramlitで作成したもの)を選択で見れるようにした")
    st.write("--- 実験の使用するカラムの重み付けをスライダーで設定できるようにした")
    st.write("- 今後の予定の確認")
    st.write("- アドバイス")

    st.header("UIデザインの変更")
    st.write("実際に動かして確認")
    st.write("- 今までの発表内容(stramlitで作成したもの)を選択で見れるようにした")
    st.write("- 実験の使用するカラムの重み付けをスライダーで設定できるようにした")

    st.header("今後の予定の確認")
    st.write("- 特徴量エンジニアリング")
    st.write("- 特徴量重要度の評価")
    st.write("- ハイパーパラメータ(C)のチューニング")
    st.write("- その他")


    st.header("特徴量エンジニアリング")
    st.write("- 目的変数(痛みの種類)の数値化の見直し")
    st.write("--- 前回、侵害T = 0 , 神経T = 1 , 不明T = 2とおいたが、上手く分類できていないのではないか")
    st.write("--- 自然においてあげるなら、侵害T = -1 , 神経T = 1 , 不明T = 0 でおく")
    st.write("--- Word2vecで数値化する")
    st.write("- 重み付けを行うカラムと値の見直し")
    st.write("--- 前回、相対的に負の相関にあるカラムを0.5倍にしていた")
    st.write("- スケーリング（データの正規化・標準化）")

    st.header("Word2vecによる数値化")
    st.write("- 文章に含まれる単語を「数値ベクトル」に変換し、その意味を把握していくという自然言語処理の手法")
    st.write("- 語句のデータを学習させ、その中から3つの痛みの種類の数値化を行うという認識で合っているか？")
    st.write("- 数値化すると多次元のベクトルに変換されると予想されるが、一次元でなくでも大丈夫か？")
    st.write("- [Word2vecの説明](https://aismiley.co.jp/ai_news/word2vec/): 参考文献")
    st.write("- [北村さんから共有していただいた動画](https://youtu.be/sK3HqLwag_w?si=VlkOHj8PZeTzUJEM): 参考文献")



    st.header("特徴量重要度の評価")
    st.write("- 前回、相関係数による特徴量の選択と重み付けの設定を行った → 強い相関が見られず、正確な選択が難しい")
    st.write("- 特徴選択：RFE（再帰的特徴消去）Recursive Feature Elimination の略")
    st.write("--- RFE：最も特徴量重要度が低いものを削除することを繰り返し、指定した特徴量の数まで消去をする手法")

    st.header("ハイパーパラメータ(C)のチューニング")
    st.write("- クロスバリデーション（交差検証）")
    st.write("--- モデル性能の評価は行ったが、ハイパーパラメータのチューニングはできるのか？")
    st.write("- グリッドサーチ")
    st.write("--- 指定された範囲の中で、すべての組み合わせを総当たりで探索して最適なパラメータを設定する手法")
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/12:13/1213_svm_parameter.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, use_container_width=True)
    st.write("--- svm.setKernel(cv2.ml.SVM_LINEAR)：カーネル関数の種類を表す")
    st.write("--- svm.setGamma(1)：LINEAR以外のカーネルの場合用いるパラメータ")
    st.write("--- svm.setC(1)：どれだけ誤分類を許容するかについてのパラメータ")
    st.write("--- svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))：終了条件")
    st.write("--- メリット：重複と漏れがない")
    st.write("--- デメリット：広い範囲の探索は厳しい")
    st.write("- ランダムサーチ")
    st.write("--- 指定された回数の中で無作為に探索して最適なパラメータを設定する手法")
    st.write("--- メリット：探索の順番がランダムであるため、広い範囲を満遍なく探索できる")
    st.write("--- デメリット：重複や最適なパラメータを見逃す場合がある")
    st.write("- ベイズ最適化")
    st.write("--- ベイズ最適化を使ったアルゴリズムによる自動探索")
    st.write("--- ガウス過程といった確率モデルを用いて過去の探索履歴を考慮して、次に探索すべきハイパラを合理的に選択する")
    st.write("- OpenCVのSVMモデルが記載されているサイトあれば教えてほしい")
    st.header("その他")
    st.write("- データ分析")
    st.write("--- 最初からデータセットを細部まで確認して特性を知る")
    st.write("- 他の機械学習モデルでの実験")
    st.write("- streamlitのUIデザインの変更")



    txt = st.text_area(
        'アドバイス', height=150
    )



elif selected == '発表履歴':
    st.title('これまでの発表内容')
    
    #データの加工方法の指定
    options = ['11月8日', '11月22日', '12月13日', '1月17日']

    # セレクトボックスを作成し、ユーザーの選択を取得
    data_processing = st.selectbox('ご覧になる発表内容の日程を選択してください', options, index = None, placeholder="選択してください")
    
    # 選択されたオプションに応じた処理
    if data_processing == "11月8日":
        st.header("11月8日")
        st.write("データから偏相関係数を求め、重み付けを行いました")
        st.write("- 訓練データ：70%,テストデータ：30%")
        st.write("- FUSIONのデータセットで実行")
        st.write("- エクセル上でカラースケールを用いて可視化")
        st.write("- 重み付け：偏相関係数の高いカラムに1.5倍、低いカラムに0.5倍で実行")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/pain_exper.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, caption='PainDETECT', use_container_width=True)

        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/bspop_exper.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, caption='BS-POP', use_container_width=True)

        st.header("実験前")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/実験前_omomi.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, use_container_width=True)

        st.header("実験後")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/picture/実験後_omomi.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, use_container_width=True)
        st.write("- 何度か実験を行ったが、実験前と変化は見られなかった")
        st.write("- 極端に大きく、小さくしたが、変化は見られなかった（若干下がったかも）")

    elif data_processing == "11月22日":
        st.header("11月22日")
        st.write("- UIデザインの変更")
        st.write("--- 実験の重み付けの指定の部分を、縦ではなく横に出力するように変更")
        st.write("--- サイドバーをプルダウン式ではなく、一覧を表示するようにした")
        st.write("--- 実験の使用するカラムの指定を全選択するようにした")
        st.write("- 目的変数(痛みの種類)を含めて相関係数を求めて重み付けを行った")

        st.header("目的変数(痛みの種類)を含めて相関係数を求めて重み付け")
        st.write("目的変数(痛みの種類)と説明変数(各質問項目)との相関係数を求め、関係性の強さを示し、結果を元に重み付けと特徴量選択をして実験を行った")
        st.write("- 相関係数の計算に用いたデータ：２つの質問表を組み合わせたもの")
        st.write("- 痛みの種類：侵害T = 0 , 神経T = 1 , 不明T = 2")
        st.markdown('#### 相関係数')
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/coeff_pain.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img, use_container_width=True)
        st.write("- 比較的に高い相関が見られなかった")
        st.write("- 相対的に見て負の相関がない、重み付けの判定がしづらい")

        st.markdown('#### 重み付け')
        st.write("- 相関係数の高いカラムに1.5倍、低いカラムに0.5倍で実行")
        st.write("- 1.5倍：P8,P12")
        st.write("- 0.5倍：P9,D10,D14,D15,D17")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_null.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="欠損値削除" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_median.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="中央値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_mean.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="平均値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_weight_knn.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="k-NN法補完" , use_container_width=True)
        st.write("結果・考察")
        st.write("- 様々な欠損値の補完データで試したが、どれも結果にばらつきがり、かつ精度が上がらなかった")

        st.markdown('#### 特徴量選択')
        st.write("- 相関係数の高いカラム、低いカラムを選択し、使用する特徴量を圧縮して実験をする")
        st.write("- 使用するカラム：P8,P9,P12,D10,D14,D15,D17")
        st.write("- 重み付けは、上記と同様に行う")
        st.write("- 0.5倍：P9,D10,D14,D15,D17")
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_null.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="欠損値削除" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_median.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="中央値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_mean.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="平均値補完" , use_container_width=True)
        img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/11:22/11_22_select_knn.png')
        # use_column_width 実際のレイアウトの横幅に合わせるか
        st.image(img,caption="k-NN法補完" , use_container_width=True)
        st.write("結果・考察")
        st.write("- どの欠損値補完データにおいても約70%ほどに収束した")
        st.write("- ほんとに適切に特徴量選択できたのか他の選択で試す必要がある")

        st.header("今後の予定")
        st.write("- 使用するカラムの指定から重み付けを行えるスライドバーの設定")
        st.write("- streamlitのプログラムのソフトコーディング化")
        st.write("- スケーリング（データの正規化・標準化）")
        st.write("- 特徴量重要度の評価")
        st.write("--- 特徴選択：RFE（再帰的特徴消去）")
        st.write("--- RFE：最も特徴量重要度が低いものを削除することを繰り返し、指定した数まで消去をする手法")

    elif data_processing == "12月13日":
        st.write("焦るな")

    elif data_processing == "1月17日":
        st.write("焦るな")


elif selected == 'データセット':
    st.title('疼痛のデータセットの表示')
    st.markdown('#### 侵害受容性疼痛')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_1")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df1 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df1)

    st.markdown('#### 神経障害性疼痛')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_2")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df2 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df2)

    st.markdown('#### 原因不明')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_3")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df3 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df3)

elif selected == 'プログラム':
    st.title('データ加工・機械学習のプログラムを表示')
    st.header('欠損値補完')
    body_1 = """
    import pandas as pd
    from sklearn.impute import KNNImputer
    from sklearn.impute import SimpleImputer
    import numpy as np

    df1 = pd.read_csv('null/peinditect/PAINDITECT.csv')

    #特定の文字列を欠損値に置き換え
    df1.replace(['#REF!', 'N/A', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    #数値データのみを抽出
    number_data = df1.select_dtypes(include=[float, int])

    class KNN:
        #最近傍法（K-Nearest Neighbors）で欠損値を5つの近傍を使用して補完
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(number_data)
        imputed_data = np.round(imputed_data).astype(int)

        df1[number_data.columns] = imputed_data
        #求められた値が0の時反映されない場合があるため、残っている欠損値をすべて0で埋める
        df1.fillna(0, inplace=True)

        det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
        det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
        det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

        det_other.to_csv('det_painditect_KNN_不明.csv', index=False)
        det_inj_pain.to_csv('det_painditect_KNN_侵害受容性疼痛.csv', index=False)
        det_neuro_pain.to_csv('det_painditect_KNN_神経障害性疼痛.csv', index=False)

    #中央値で欠損値を補完
    class median:
        imputer = SimpleImputer(strategy='median')

        imputed_data = imputer.fit_transform(number_data)

        imputed_data = np.round(imputed_data).astype(int)

        df1[number_data.columns] = imputed_data
        df1.fillna(0, inplace=True)

        det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
        det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
        det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

        det_other.to_csv('det_painditect_median_不明.csv', index=False)
        det_inj_pain.to_csv('det_painditect_median_侵害受容性疼痛.csv', index=False)
        det_neuro_pain.to_csv('det_painditect_median_神経障害性疼痛.csv', index=False)

    #平均値で欠損値を補完
    class mean:
        imputer = SimpleImputer(strategy='mean')

        imputed_data = imputer.fit_transform(number_data)

        imputed_data = np.round(imputed_data).astype(int)

        df1[number_data.columns] = imputed_data
        df1.fillna(0, inplace=True)

        det_other = df1[~df1['痛みの種類'].str.match(r'^侵害受容性疼痛$|^神経障害性疼痛$', na=False)]
        det_inj_pain = df1[df1['痛みの種類'].str.match(r'^侵害受容性疼痛$', na=False)]
        det_neuro_pain = df1[df1['痛みの種類'].str.match(r'^神経障害性疼痛$', na=False)]

        det_other.to_csv('det_painditect_mean_不明.csv', index=False)
        det_inj_pain.to_csv('det_painditect_mean_侵害受容性疼痛.csv', index=False)
        det_neuro_pain.to_csv('det_painditect_mean_神経障害性疼痛.csv', index=False)

    KNN()
    median()
    mean()
    """
    st.code(body_1, language="python")

    st.header('特徴量増量')
    body_2 = """
    import pandas as pd

    # CSVファイルのリストを指定
    csv_files = ['null/fusion/questionnaire_fusion_missing_侵害受容性疼痛.csv',
                'null/fusion/questionnaire_fusion_missing_神経障害性疼痛.csv',
                'null/fusion/questionnaire_fusion_missing_不明.csv',
                '欠損値補完/FUSION/det_KNN_侵害受容性疼痛.csv',
                '欠損値補完/FUSION/det_KNN_神経障害性疼痛.csv',
                '欠損値補完/FUSION/det_KNN_不明.csv',
                '欠損値補完/FUSION/det_mean_侵害受容性疼痛.csv',
                '欠損値補完/FUSION/det_mean_神経障害性疼痛.csv',
                '欠損値補完/FUSION/det_mean_不明.csv',
                '欠損値補完/FUSION/det_median_侵害受容性疼痛.csv',
                '欠損値補完/FUSION/det_median_神経障害性疼痛.csv',
                '欠損値補完/FUSION/det_median_不明.csv',]

    # 列名を指定
    column1 = '②'
    column2 = '⑥'
    new_column1 = '痺れ'
    column3 = '③'
    column4 = '⑦'
    new_column2 = '少しの痛み'
    column5 = '④.1'
    column6 = '④.2'
    new_column3 = '機嫌'
    column7 = '⑥.1'
    column8 = '⑧.1'
    new_column4 = 'しつこさ'

    # 各CSVファイルに対して処理を行う
    for csv_file in csv_files:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_file)

        # 列の掛け算をして新しい列を追加する
        df[new_column1] = df[column1] * df[column2]
        df[new_column2] = df[column3] * df[column4]
        df[new_column3] = df[column5] * df[column6]
        df[new_column4] = df[column7] * df[column8]

        # 新しいCSVファイルとして保存する（元のファイル名に "_modified" を追加）
        output_csv_file_path = csv_file.replace('.csv', '_newroc.csv')
        df.to_csv(output_csv_file_path, index=False)
    """
    st.code(body_2, language="python")
    
    st.header('svm実装')
    body_3 = """
    TEST_DATA_RATIO = 0.3
    SAVE_TRAINED_DATA_PATH = "svm_data.xml"

    # csvファイルの読み込み
    df = pd.read_csv(uploaded_file, encoding = 'utf-8')

    # カラムと重みの値を取得
    columns = df["columns"].tolist()
    weights = df["weights"].tolist()
    
    # データの分割
    df_nociceptive_train, df_nociceptive_test = train_test_split(
        df1[columns], test_size=TEST_DATA_RATIO, random_state=None
        )
    df_neuronociceptive_train, df_neuronociceptive_test = train_test_split(
        df2[columns], test_size=TEST_DATA_RATIO, random_state=None
        )
    df_unknown_train, df_unknown_test = train_test_split(
        df3[columns], test_size=TEST_DATA_RATIO, random_state=None
        )

    # 重みを適用して特徴量を調整（訓練データの場合）
    df_nociceptive_train_weighted = df_nociceptive_train.mul(weights, axis=1)
    df_nociceptive_test_weighted = df_nociceptive_test.mul(weights, axis=1)

    # トレーニングデータとラベルの作成
    datas = np.vstack(
        [
            df_nociceptive_train.values,
            df_neuronociceptive_train.values,
            df_unknown_train.values,
            ]
            ).astype(np.float32)
    
    labels1 = np.full(len(df_nociceptive_train), 1, np.int32)
    labels2 = np.full(len(df_neuronociceptive_train), 2, np.int32)
    labels3 = np.full(len(df_unknown_train), 3, np.int32)
    labels = np.concatenate([labels1, labels2, labels3]).astype(np.int32)
    
    # SVMモデルの作成とトレーニング
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setGamma(1)
    svm.setC(1)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 1500, 1.0e-06))
    svm.train(datas, cv2.ml.ROW_SAMPLE, labels)
    
    # モデルを保存
    svm.save(SAVE_TRAINED_DATA_PATH)
    
    test_datas = np.vstack(
        [
        df_nociceptive_test.values,
        df_neuronociceptive_test.values,
        df_unknown_test.values,
        ]
        ).astype(np.float32)
    
    test_labels1 = np.full(len(df_nociceptive_test), 1, np.int32)
    test_labels2 = np.full(len(df_neuronociceptive_test), 2, np.int32)
    test_labels3 = np.full(len(df_unknown_test), 3, np.int32)
    
    test_labels = np.concatenate([test_labels1, test_labels2, test_labels3]).astype(
        np.int32
        )
    
    # # データの標準化
    # scaler = StandardScaler()
    # datas = scaler.fit_transform(datas)
    # test_datas = scaler.transform(test_datas)
    
    # # 交差検証の実行
    # cross_val_scores = cross_val_score(svm, datas, labels, cv=5)
    # print("Cross-Validation Scores:", cross_val_scores)
    # print("Mean Cross-Validation Score:", cross_val_scores.mean())
    
    svm = cv2.ml.SVM_load(SAVE_TRAINED_DATA_PATH)
    _, predicted = svm.predict(test_datas)
    
    confusion_matrix = np.zeros((3, 3), dtype=int)
    
    for i in range(len(test_labels)):
        index1 = test_labels[i] - 1
        index2 = predicted[i][0] - 1
        confusion_matrix[int(index1)][int(index2)] += 1
        
    st.write("confusion matrix")
    st.table(confusion_matrix)

    score = np.sum(test_labels == predicted.flatten()) / len(test_labels)
        
    st.write("正答率:", score*100, "%")
        
    # 感度と特異度の計算
    sensitivity = np.zeros(3)
    specificity = np.zeros(3)
    
    for i in range(3):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = np.sum(confusion_matrix) - (TP + FN + FP)
        
        sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0
        
    # 感度と特異度の表示
    st.write("感度と特異度")
    st.write("（疼痛1:侵害受容性疼痛,疼痛2:神経障害性疼痛,疼痛3:不明）")
    for i in range(3):
        st.write(f"疼痛 {i+1}: 感度 = {sensitivity[i]:.4f}, 特異度 = {specificity[i]:.4f}")
    """
    st.code(body_3, language="python")

elif selected == 'データ分析':
    st.title('データ分析')
    st.markdown('偏相関係数の評価を表示')
    st.markdown('#### PainDETECT')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_1")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df8 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        # データフレーム内の2つの変数 (target1 と target2) の間で、他の変数 (control_vars) の影響を取り除いた偏相関係数を計算
        def calculate_partial_correlation(df, target1, target2, control_vars):
            # 回帰モデルの作成と実測値から予測値の残差の計算
            model1 = LinearRegression().fit(df[control_vars], df[target1])
            residuals_target1 = df[target1] - model1.predict(df[control_vars])

            model2 = LinearRegression().fit(df[control_vars], df[target2])
            residuals_target2 = df[target2] - model2.predict(df[control_vars])

            # 残差間の相関を計算
            partial_corr = np.corrcoef(residuals_target1, residuals_target2)[0, 1]
            return partial_corr

        # 新しい変数リストの作成（P1からP13とD1からD18）
        variables = [f'P{i}' for i in range(1, 14)]
        partial_corr_matrix = pd.DataFrame(index=variables, columns=variables)

        # ループを使って全てのペアで偏相関を計算
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                target1 = variables[i]
                target2 = variables[j]
                
                # target1とtarget2を除いた他の変数を制御変数とする
                control_vars = [v for v in variables if v != target1 and v != target2]
                
                # 各ペアの偏相関を計算
                partial_corr = calculate_partial_correlation(df8, target1, target2, control_vars)
                
                # 行列に対称的に格納
                partial_corr_matrix.loc[target1, target2] = partial_corr
                partial_corr_matrix.loc[target2, target1] = partial_corr

        # 対角成分に1を設定（自己相関）
        np.fill_diagonal(partial_corr_matrix.values, 1)

        pd.set_option('display.max_columns', 50)

        # 数値を小数第3位まで丸める
        pd.options.display.float_format = '{:.2f}'.format

        # 現在の日時を取得してファイル名に追加
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_path = f'/Users/iwasho_0225/Desktop/workspace/pain_experiment/相関係数/PainDETECT/partial_{timestamp}.csv'

        # CSVファイルとして出力
        partial_corr_matrix.to_csv(output_path, float_format="%.3f")

        # 完了メッセージの表示
        st.success(f"CSVファイルが '{output_path}' に保存されました。")

    st.markdown('#### BS-POP')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_2")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df8 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        # データフレーム内の2つの変数 (target1 と target2) の間で、他の変数 (control_vars) の影響を取り除いた偏相関係数を計算
        def calculate_partial_correlation(df, target1, target2, control_vars):
            # 回帰モデルの作成と実測値から予測値の残差の計算
            model1 = LinearRegression().fit(df[control_vars], df[target1])
            residuals_target1 = df[target1] - model1.predict(df[control_vars])

            model2 = LinearRegression().fit(df[control_vars], df[target2])
            residuals_target2 = df[target2] - model2.predict(df[control_vars])

            # 残差間の相関を計算
            partial_corr = np.corrcoef(residuals_target1, residuals_target2)[0, 1]
            return partial_corr

        # 新しい変数リストの作成（P1からP13とD1からD18）
        variables = [f'D{i}' for i in range(1, 19)]
        partial_corr_matrix = pd.DataFrame(index=variables, columns=variables)

        # ループを使って全てのペアで偏相関を計算
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                target1 = variables[i]
                target2 = variables[j]
                
                # target1とtarget2を除いた他の変数を制御変数とする
                control_vars = [v for v in variables if v != target1 and v != target2]
                
                # 各ペアの偏相関を計算
                partial_corr = calculate_partial_correlation(df8, target1, target2, control_vars)
                
                # 行列に対称的に格納
                partial_corr_matrix.loc[target1, target2] = partial_corr
                partial_corr_matrix.loc[target2, target1] = partial_corr

        # 対角成分に1を設定（自己相関）
        np.fill_diagonal(partial_corr_matrix.values, 1)

        pd.set_option('display.max_columns', 50)

        # 数値を小数第3位まで丸める
        pd.options.display.float_format = '{:.2f}'.format

        # 現在の日時を取得してファイル名に追加
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_path = f'/Users/iwasho_0225/Desktop/workspace/pain_experiment/相関係数/BSPOP/partial_{timestamp}.csv'

        # CSVファイルとして出力
        partial_corr_matrix.to_csv(output_path, float_format="%.3f")

        # 完了メッセージの表示
        st.success(f"CSVファイルが '{output_path}' に保存されました。")

    st.markdown('#### FUSION')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_3")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df8 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        def calculate_partial_correlation(df, target1, target2, control_vars):
            model1 = LinearRegression().fit(df[control_vars], df[target1])
            residuals_target1 = df[target1] - model1.predict(df[control_vars])

            model2 = LinearRegression().fit(df[control_vars], df[target2])
            residuals_target2 = df[target2] - model2.predict(df[control_vars])

            partial_corr = np.corrcoef(residuals_target1, residuals_target2)[0, 1]
            return partial_corr

        # 新しい変数リストの作成（P1からP13とD1からD18）
        variables = [f'P{i}' for i in range(1, 14)] + [f'D{i}' for i in range(1, 19)]
        partial_corr_matrix = pd.DataFrame(index=variables, columns=variables)

        # ループを使って全てのペアで偏相関を計算
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                target1 = variables[i]
                target2 = variables[j]
                
                # target1とtarget2を除いた他の変数を制御変数とする
                control_vars = [v for v in variables if v != target1 and v != target2]
                
                # 各ペアの偏相関を計算
                partial_corr = calculate_partial_correlation(df8, target1, target2, control_vars)
                
                # 行列に対称的に格納
                partial_corr_matrix.loc[target1, target2] = partial_corr
                partial_corr_matrix.loc[target2, target1] = partial_corr

        # 対角成分に1を設定（自己相関）
        np.fill_diagonal(partial_corr_matrix.values, 1)

        pd.set_option('display.max_columns', 50)

        # 数値を小数第3位まで丸める
        pd.options.display.float_format = '{:.2f}'.format

        # 現在の日時を取得してファイル名に追加
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_path = f'/Users/iwasho_0225/Desktop/workspace/pain_experiment/相関係数/FUSION/partial_{timestamp}.csv'

        # CSVファイルとして出力
        partial_corr_matrix.to_csv(output_path, float_format="%.3f")

        # 完了メッセージの表示
        st.success(f"CSVファイルが '{output_path}' に保存されました。")

    st.markdown('#### 相関係数のCSVファイル参照')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_4")

    if uploaded_file:
        # CSVファイルの読み込み
        df = pd.read_csv(uploaded_file)
        
        # データを小数第3位まで丸める
        df = df.round(3)
        
        # データフレームとして出力
        st.dataframe(df)

elif selected == '実験結果の保存':
    st.title('実験結果のログを表示')
    st.markdown('#### ログのCSVファイルを参照')
    # ファイルアップローダーの準備
    uploaded_file = st.file_uploader("CSVファイルのアップロード", type="csv", key="unique_key_1")
    
    # uploadファイルが存在するときだけ、csvファイルの読み込み
    if uploaded_file :
        df1 = pd.read_csv(uploaded_file, encoding = 'utf-8')

        st.markdown('#### データセット')
        st.dataframe(df1)

else:
    st.title('質問表を表示')
    st.markdown('#### PainDETECT')
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/painditect.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='参考文献：https://gunma-pt.com/wp-content/uploads/2015/03/paindetect.pdf', use_container_width=True)

    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/painDETECT-Q.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='参考文献：https://www.researchgate.net/figure/The-painDETECT-Questionnaire-Japanese-version-PDQ-J-doi_fig3_257465057', use_container_width=True)

    st.markdown('#### BS-POP')
    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_医師.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, use_container_width=True)

    img = Image.open('/Users/iwasho_0225/Desktop/workspace/pain_experiment/質問表/bspop_患者.png')
    # use_column_width 実際のレイアウトの横幅に合わせるか
    st.image(img, caption='参考文献：http://www.onitaiji.com/spine/evaluation/0.pdf', use_container_width=True)




