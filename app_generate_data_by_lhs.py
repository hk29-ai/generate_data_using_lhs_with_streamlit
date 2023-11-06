import streamlit as st
from itertools import product
import lhsmdu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 14 # グラフの基本フォントサイズの設定
st.set_option('deprecation.showPyplotGlobalUse', False)
import datetime
now = datetime.datetime.now()
now = now.strftime("%y%m%d")


def generate_doe_func():
    st.write('はじめに因子名を入力します。次に、各因子の値を入力します。')

    # 最初に列名を入力
    column_names = st.text_input('作成する因子名を複数入力してください（カンマ区切り。例：A,B, C）')
    columns = [name.strip() for name in column_names.split(',')]

    # 空の辞書を作成
    input_params = {col: [] for col in columns}

    # 各列名に対するデータ入力
    for col in columns:
        input_data = st.text_input(f'因子{col}のデータを入力してください（カンマ区切り。例：10, 15, 20, 25）')
        input_params[col] = [x.strip() for x in input_data.split(',')]

    # データが入力されると随時反映
    if any(input_params):
        # テーブルの表示
        st.write('入力されている値', input_params)

        if st.button('実行'):
            # 各キーのバリューの組み合わせを生成
            combinations = list(product(*input_params.values()))

            # データフレームを作成
            df = pd.DataFrame(combinations, columns=input_params.keys())
            st.write(f'水準数は{len(df)}です。', df)

            # ダウンロードボタンの追加
            csv_file = df.to_csv(index=False)

            download_button = st.download_button(
                label = "データをダウンロード",
                data = csv_file,
                file_name = "doe.csv",
                key = "download_button"
            )

class RandomSampler:
    # インスタンスの生成
    def __init__(self, factors, sampling_num, save_file_name, plot_color):
        self.factors = factors # 入力データ
        self.factors_column_names = list(factors.keys()) # 因子名をリスト化
        self.sampling_num = sampling_num # 生成するサンプル数
        self.save_file_name = save_file_name # 保存するファイル名
        self.plot_color = plot_color # グラフにプロットする点の色

    # データ生成
    def generate_random_data(self, random_data):
        # データの逆正規化
        fixed_data_np = self._normalize_data(random_data)
        
        # pandasデータフレームにする
        random_df = pd.DataFrame(fixed_data_np.T, columns=self.factors_column_names)
        
        # csvファイルへ出力する
        random_df.to_csv(f"{self.save_file_name}.csv", sep=",", index=False, encoding="utf-8")
        
        # グラフへプロット
        self.plot_matrix_scatter(random_df)
        
        return random_df

    # データの逆正規化。生成されたデータは区間0～1の乱数データのため、指定した最小値～最大値の区間へ変換する
    def _normalize_data(self, data):
        for i, key in enumerate(self.factors_column_names):
            my_min, my_max = self.factors[key]
            data[i] = ((my_max - my_min) * data[i] + my_min)
        return np.array(data)

    # 行列散布図
    def plot_matrix_scatter(self, df):
        sns.set(style="ticks", font_scale=1.2, palette=self.plot_color, color_codes=True)
        ax = sns.pairplot(df, diag_kind="hist")
        ax.fig.suptitle(self.save_file_name)
        ax.fig.subplots_adjust(top=0.9)
        st.pyplot(ax)

# ラテン超方格法によるデータ生成
class LatinHypercube(RandomSampler):
    def __init__(self, factors, sampling_num, save_file_name, plot_color):
        super().__init__(factors, sampling_num, save_file_name, plot_color)

    def generate_samples(self):
        latin_hypercube = lhsmdu.sample(len(self.factors), self.sampling_num)
        return self.generate_random_data(latin_hypercube)


def generate_lhs_func():

    # 最初に列名を入力
    column_names = st.text_input('作成する因子名を複数入力してください（カンマ区切り。例：A,B, C）')
    columns = [name.strip() for name in column_names.split(',')]
    
    # 空の辞書を作成
    input_params = {col: [] for col in columns}

    # 各列名に対するデータ入力
    st.write('因子と上下限を辞書で作成')
    for col in columns:
        input_data = st.text_input(f'因子{col}の下限と上限データを入力してください（カンマ区切り。例：100, 200）')
        input_params[col] = [x.strip() for x in input_data.split(',')]

    # データが入力されると随時反映
    if any(input_params):
    
        st.write(input_params)
        
        # 作成するサンプリング数
        sampling_num = st.number_input('生成するサンプル数を入力して下さい。直接手入力できます。',
                                       min_value=100, value=200, step=10, format="%d")
        st.write('設定されたサンプル数：', sampling_num)
        
        if st.button('実行'):
            latin_hypercube  = LatinHypercube(input_params, sampling_num, f'{now}_latin_hypercube', 'autumn')
            df2 = latin_hypercube.generate_samples()
            st.write(df2)

            # ダウンロードボタンの追加
            csv_file = df2.to_csv(index=False)

            download_button = st.download_button(
                label = "データをダウンロード",
                data = csv_file,
                file_name = "lhs.csv",
                key = "download_button"
            )

def main():
    st.set_page_config(
        page_title = "水準表を作成するwebアプリ",
        page_icon = "🧊",
        layout = "centered",
        initial_sidebar_state = "expanded"
    )

    st.subheader('水準表を作成するwebアプリ')

    # 背景色を変更した説明文
    description = """
    <div style="background-color: #87CEFA; padding: 10px; border-radius: 10px;">
        <p style="font-weight: normal;">すべてをリセットしたい場合はブラウザをリロード下さい</p>
    </div>
    """
    st.markdown(description, unsafe_allow_html=True)

    read_method = st.radio(label = 'データ生成の手法を選択ください',
                           options = ('DOE（格子状にデータ生成）', 'LHS（生成数を指定して均等にデータ生成）'),
                           index = 0,
                           horizontal = True,
                           )
    
    if read_method == 'DOE（格子状にデータ生成）':
        generate_doe_func()
    else:
        generate_lhs_func()

if __name__ == "__main__":
    main()
    
