import streamlit as st
import lhsmdu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 14 # グラフの基本フォントサイズの設定
st.set_option('deprecation.showPyplotGlobalUse', False)
from io import BytesIO

st.title("ラテン超方格法によるデータの生成")

# 作成するサンプリング数
sampling_num = st.number_input('生成するサンプル数を入力して下さい。直接手入力できます。', step=1, format="%d")
st.write('設定されたサンプル数：', sampling_num)

# 因子と上下限を辞書で作成
factor_dict = {
    "height":(50, 200),
    "width":(0.06, 0.1),
    "density":(1e15, 9e15),
    "temp":(-50, 250)
}


# 因子名をリストで作成
column_list = []
for k in factor_dict.keys():
    column_list.append(k)


# 乱数データ区間0～1のを指定した最小値～最大値の区間へ変換する関数
def adjust_data_func(_column_list, _random_np):
    for i, key in enumerate(_column_list):
        _min, _max = factor_dict[key]
        _random_np[i] = ((_max - _min) * _random_np[i] + _min)
    _fixed_randam_np = np.array(_random_np)

    return _fixed_randam_np


# ラテン超方格法
def latin_hypercube_func(_factors, _column_list, _sampling_num):
    # ライブラリでデータを生成
    _data = lhsmdu.sample(len(_factors), _sampling_num)
    
    # 作成したデータは0～1区間のため、指定した最小値～最大値の区間へ変換する
    _fixed_data_np = adjust_data_func(_column_list, _data)
    
    # pandasデータフレーム形式へ変換
    _df = pd.DataFrame(_fixed_data_np.T)
    
    # 列名を設定
    _df.columns = _column_list
    
    return _df
    

# 行列散布図を作成する関数
def plot_matrix_scatter_func(_label, _df, _color):
    sns.set(style = "ticks", font_scale = 1.2, palette = _color, color_codes = True)
    ax = sns.pairplot(_df,
                      diag_kind="hist"
                      )
    ax.fig.suptitle(_label) #, fontsize=12)
    ax.fig.subplots_adjust(top = 0.9)

    return ax

# csv形式へ変換する関数
def convert_df(_df):
    return _df.to_csv(index=False).encode('utf-8')


if st.button('実行', key='my_button1', help=''):
    # データの生成
    df = latin_hypercube_func(factor_dict, column_list, sampling_num)
    st.write('生成されたデータ', df)
    
    # データを可視化のためにグラフを作成
    fig = plot_matrix_scatter_func("latin_hypercube", df, "autumn")
    st.pyplot(fig)
    
    # データをcsvへ保存
    _csv = convert_df(df)
    st.download_button(
        label = "CSVファイルのダウンロード",
        data = _csv,
        file_name = 'latin_hypercube.csv',
        mime = 'text/csv'
    )
    