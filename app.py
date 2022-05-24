""" streamlit_demo
streamlitでIrisデータセットの分析結果を Web アプリ化するモジュール
"""

#

import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 
import graphviz
import plotly.graph_objects as go
# irisデータセットでテストする
from sklearn.datasets import load_iris
# 決定木で分類してみる
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sns.set()


def make_iris_df():
    """ Irisデータフレーム作成関数
    Irisデータセットをデータフレームとして返す関数
    Args:
    
    Returns:
        df(pd.DataFrame): Irisデータセットのデータフレーム
    """
    dataset = load_iris()
    df = pd.DataFrame(dataset.data)
    # 変数名を列名に代入
    df.columns = dataset.feature_names
    # 目的変数を設定
    df["species"] = dataset.target

    return df


def st_display_df(df: pd.DataFrame):
    """ データフレーム表示関数
    streamlitでデータフレームを表示する関数
    Args:
        df(pd.DataFrame): stで表示するデータフレーム
    
    Returns:
    """
    # 表示する行数を選択
    row_size = st.number_input(
        "表示する行数を選択してください(下スクロールで追加行が表示されます)。",
        min_value=10,
        max_value=50,
        value=10,
        step=10
    )
    # データフレームを表示
    st.dataframe(df.head(row_size))


def st_display_pairplot(df: pd.DataFrame):
    """ ペアプロットをstに表示する関数
    streamlitでsns.pairplotを表示する関数
    Args:
        df(pd.DataFrame): pairplotを作成するデータ
    Returns:
    """
    # ペアプロットを作成
    fig = plt.figure()
    fig = sns.pairplot(df, hue="species")
    # stで表示
    st.pyplot(fig)


def st_display_plotly(df: pd.DataFrame):
    """plotlyグラフをstに表示する関数
    streamlitでplotlyグラフを表示する関数
    Args:
        df(pd.DataFrame): plotlyグラフを作成するデータ
    Returns:
    """
    # plotlyグラフを適当に作成
    fig = go.Figure(data=[
        go.Scatter(
            x=df.loc[df["species"]==0, "petal length (cm)"],
            y=df.loc[df["species"]==0, "petal width (cm)"],
            name="setosa",
            mode="markers"),
        go.Scatter(
            x=df.loc[df["species"]==1, "petal length (cm)"],
            y=df.loc[df["species"]==1, "petal width (cm)"],
            name="versicolor",
            mode="markers"),
        go.Scatter(
            x=df.loc[df["species"]==2, "petal length (cm)"],
            y=df.loc[df["species"]==2, "petal width (cm)"],
            name="virginica",
            mode="markers"),
    ])
    # stに表示
    st.plotly_chart(fig, user_container_width=True)


def ml_dtree(
    X: pd.DataFrame,
    y: pd.Series) -> list:
    """ 決定木で学習、予測を行う関数
    Irisデータセット全体で学習し、学習データの予測値を返す関数
    Args:
        X(pd.DataFrame): 説明変数郡
        y(pd.Series): 目的変数
    
    Returns:
        List: [モデル, 学習データを予測した予測値, accuracy]のリスト
    """
    # 学習
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = accuracy_score(y, pred)

    return [clf, pred, score]


def st_display_dtree(clf):
    """決定木可視化関数
    streamlitでDtreeVizによる決定木を可視化する関数
    Args:
        clf(sklearn.DecisionTreeClassifier): 学習済みモデル
    Return:
    """
    # graphvizで決定木を可視化
    dot = tree.export_graphviz(clf, out_file=None)
    # stで表示する
    st.graphviz_chart(dot)


def st_file_uploader() -> pd.DataFrame:
    """ ファイルアップロードを受け付ける関数
    streamlitでファイルアップロードを受け付ける関数
    Args:
    Returns:
        pd.DataFrame: 予測用のデータ
    """
    uploaded_file = st.file_uploader(
        "予測対象となる CSV ファイルをアップロードしてください。",
        type="csv",
        accept_multiple_files=False
    )
    if uploaded_file:
        pred_df = pd.read_csv(uploaded_file)
        return pred_df
    else:
        return pd.DataFrame()


def ml_pred(clf, X: pd.DataFrame) -> np.array:
    """ 予測用関数
    与えられたデータに対して予測値を返す関数
    Args:
        clf(sklearn.tree.DecisionTreeClassifier): 学習済みモデル
        X(pd.DataFrame): 予測対象データ
    Returns:
        np.array: 予測結果
    """
    pred = clf.predict(X)
    return pred


if __name__ == "__main__":
    df = make_iris_df()

    # stのタイトル表示
    st.title("Iris データセットで Streamlit をお試し")

    # データフレーム表示
    st.markdown(r"## Iris データの詳細")
    st_display_df(df)
    st.text("")
    
    # pairplot表示
    st.markdown(r"## Iris データの Pair Plot")
    st_display_pairplot(df)
    st.text("")

    # plotlyグラフ表示
    st.markdown(r"## Petal length と Petal width の plotly グラフ")
    st_display_plotly(df)
    st.text("")

    # 学習
    st.markdown(r"## 決定木で学習")
    
    X = df.drop("species", axis=1)
    y = df["species"]
    clf, pred, score = ml_dtree(X, y)

    st.text(f"精度（accuracy）は {score} でした。")
    st.text("")

    # 決定木の可視化
    st.markdown(r"## 学習した決定木")
    st_display_dtree(clf)
    st.text("")

    # 予測対象ファイルの受付
    st.markdown(r"## 予測用ファイルをアップロードしてください。")
    pred_df = st_file_uploader()

    # 予測対象の予測値を算出
    if len(pred_df):
        st.text("予測結果を確認してください（以下の pred カラムが予測値です）。")
        pred_df["pred"] = ml_pred(clf, pred_df)
        st.dataframe(pred_df)