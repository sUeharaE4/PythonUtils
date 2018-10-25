import pandas as pd
import numpy as np


class CleansingUtils:

    def __init__(self):
        """


        """

    @classmethod
    def fill_nan_mean(cls, orig_data, col_name, is_int=False):
        """
        NaN をすでにある値の平均値で埋める

        Parameters
        ----------
        orig_data : pandas の DataFrame
            元データ
        col_name : str
            対称のカラム名
        is_int : bool
            本来はintの予定だけどfloatになっていて、返却値をintにしたい場合True。

        Returns
        -------
        fill_data : pandas の DataFrame
            NaN を埋めたデータ
        """
        fill_data = orig_data
        tmp_data = orig_data[col_name].fillna(orig_data[col_name].mean())
        fill_data[col_name] = tmp_data

        if is_int:
            fill_data[col_name] = fill_data[col_name].astype(int)
        return fill_data

    @classmethod
    def fill_nan_median(cls, orig_data, col_name, is_int=False):
        """
        NaN をすでにある値の中央値で埋める

        Parameters
        ----------
        orig_data : pandas の DataFrame
            元データ
        col_name : str
            対称のカラム名
        is_int : bool
            本来はintの予定だけどfloatになっていて、返却値をintにしたい場合True。

        Returns
        -------
        fill_data : pandas の DataFrame
            NaN を埋めたデータ
        """
        fill_data = orig_data
        tmp_data = orig_data[col_name].fillna(orig_data[col_name].median())
        fill_data[col_name] = tmp_data

        if is_int:
            fill_data[col_name] = fill_data[col_name].astype(int)
        return fill_data

    @classmethod
    def fill_nan_mode(cls, orig_data, col_name, is_int=False):
        """
        NaN をすでにある値の最頻値で埋める

        Parameters
        ----------
        orig_data : pandas の DataFrame
            元データ
        col_name : str
            対称のカラム名
        is_int : bool
            本来はintの予定だけどfloatになっていて、返却値をintにしたい場合True。

        Returns
        -------
        fill_data : pandas の DataFrame
            NaN を埋めたデータ
        """
        fill_data = orig_data
        tmp_data = orig_data[col_name].fillna(orig_data[col_name].median())
        fill_data[col_name] = tmp_data

        if is_int:
            fill_data[col_name] = fill_data[col_name].astype(int)
        return fill_data

    @classmethod
    def fill_nan_range(cls, orig_data, col_name, seed=0, is_int=False):
        """
        NaN をすでにある値の範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas の DataFrame
            元データ
        col_name : str
            対称のカラム名
        seed : int
            シード。指定したければどうぞ。
        is_int : bool
            本来はintの予定だけどfloatになっていて、返却値をintにしたい場合True。

        Returns
        -------
        fill_data : pandas の DataFrame
            NaN を埋めたデータ
        """
        fill_data = orig_data
        np.random.seed(seed)
        # 最大最小とその幅を取得
        data_max = orig_data[col_name].max()
        data_min = orig_data[col_name].min()
        data_range = data_max - data_min
        data_len = len(orig_data[col_name])

        # 乱数生成
        rand_data = data_min + np.random.rand(data_len) * data_range

        # NaN は 1, NaN 以外は 0 として扱うため、isnull 判定
        miss_data = orig_data[col_name].isnull()
        # NaN だったところのみ乱数が残り、元データがあった部分は0へ
        rand_data = rand_data * miss_data
        # 元データの NaN は0に変換してから乱数を加える
        fill_data[col_name] = orig_data[col_name].fillna(0) + rand_data

        if is_int:
            fill_data[col_name] = fill_data[col_name].astype(int)
        return fill_data

    @classmethod
    def fill_nan_list(cls, orig_data, col_name, from_list, weights=None, seed=0):
        """
        NaN をlistの範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas の DataFrame
            元データ
        col_name : str
            対称のカラム名
        from_list : list
            ランダムに抽出したい値のlist
        weights : list
            抽出時に重みづけしたい場合に指定。from_listと同じsizeで。
        seed : int
            シード。指定したければどうぞ。

        Returns
        -------
        fill_data : pandas の DataFrame
            NaN を埋めたデータ
        """
        fill_data = orig_data
        np.random.seed(seed)
        # 要素数を取得
        data_len = len(from_list)
        # 重みが与えられていない場合はすべて等しくする
        # TODO assert入れてほしい。要素数で
        if weights is None:
            tmp_list = np.ones(data_len) / data_len
            weights = tmp_list.tolist()

        # ランダムな値の抽出
        rand_data = np.random.choice(from_list, len(orig_data[col_name]), p=weights)

        # NaN は 1, NaN 以外は 0 として扱うため、isnull 判定
        # NaN だったところのみ乱数が残り、元データがあった部分は0へ
        miss_data = orig_data[col_name].isnull()
        rand_data = rand_data * miss_data
        # 元データの NaN は0に変換してから乱数を加える
        fill_data[col_name] = orig_data[col_name].fillna(0) + rand_data
        return fill_data

    @classmethod
    def fill_nan_range_date(cls, orig_data, col_name, seed=0, date_fmt=None):
        """
        NaN をすでにある値の範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas の DataFrame
            元データ
        col_name : str
            対称のカラム名
        seed : int
            シード。指定したければどうぞ。
        date_fmt : str
            NaNを埋めるついでにdatetime型に変換したい場合はフォーマットを指定する。

        Returns
        -------
        fill_data : pandas の DataFrame
            NaN を埋めたデータ
        """
        fill_data = orig_data

        if date_fmt is not None:
            fill_data[col_name] = pd.to_datetime(fill_data[col_name], format=date_fmt)
        np.random.seed(seed)
        # 最大最小とその幅を取得
        data_max = orig_data[col_name].max()
        data_min = orig_data[col_name].min()
        start_date = data_min.value // 10 ** 9
        end_date = data_max.value // 10 ** 9
        data_len = len(orig_data[col_name])

        # 乱数生成
        rand_data = pd.to_datetime(
            np.random.randint(start_date, end_date, data_len), unit='s'
        )

        # NaN は 1, NaN 以外は 0 として扱うため、isnull判定
        miss_data = orig_data[col_name].isnull()
        # NaN だったところのみ乱数が残り、元データがあった部分は0へ
        rand_data = rand_data.map(pd.Timestamp.timestamp).astype(int) * miss_data
        # 元データのNaTを0に変換してから乱数を加える
        fill_data[col_name] = fill_data[col_name].replace(pd.NaT, pd.to_datetime('1970-01-01 00:00:00'))
        fill_data[col_name] = fill_data[col_name].map(pd.Timestamp.timestamp).astype(int) + rand_data
        # 日付型に変換
        fill_data[col_name] = pd.to_datetime(fill_data[col_name] * 10 ** 9)
        # 元のフォーマットにして返却。フォーマットがなければpandasのデフォルト
        fill_data[col_name] = fill_data[col_name].dt.strftime(date_fmt)

        return fill_data