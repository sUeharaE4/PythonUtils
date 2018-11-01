import pandas as pd
import numpy as np
from logging import getLogger, INFO, StreamHandler, Formatter


class CleansingUtils:

    def __init__(self, log_level=INFO):
        """
        初期化関数

        Parameters
        ----------
        log_level : logging.INFO or DEBUG
            コンソールに処理状況を出力したい場合のみDEBUGを指定
        """
        # logger = getLogger(__name__)
        # handler = StreamHandler()
        # handler.setLevel(log_level)
        # logger.setLevel(log_level)
        # logger.addHandler(handler)
        # formatter = Formatter('%(asctime)s:%(lineno)d:%(funcName)s:%(message)s')
        # handler.setFormatter(formatter)
        # logger.propagate = False

    @classmethod
    def fill_nan_mean(cls, orig_data, col_name, cast_type=None):
        """
        NaN をすでにある値の平均値で埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
            元データ
        col_name : str
            対称のカラム名
        cast_type : str
            NaNがFloatなのでint等に変換したい場合は文字列で指定する。

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data
        tmp_data = orig_data[col_name].fillna(orig_data[col_name].mean())
        fill_data[col_name] = tmp_data

        if cast_type is not None:
            cls.__cast_int(fill_data, col_name, cast_type)
        return fill_data

    @classmethod
    def fill_nan_median(cls, orig_data, col_name, cast_type=None):
        """
        NaN をすでにある値の中央値で埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
            元データ
        col_name : str
            対称のカラム名
        cast_type : str
            NaNがFloatなのでint等に変換したい場合は文字列で指定する。

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data
        tmp_data = orig_data[col_name].fillna(orig_data[col_name].median())
        fill_data[col_name] = tmp_data

        if cast_type is not None:
            cls.__cast_int(fill_data, col_name, cast_type)
        return fill_data

    @classmethod
    def fill_nan_mode(cls, orig_data, col_name, cast_type=None):
        """
        NaN をすでにある値の最頻値で埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
            元データ
        col_name : str
            対称のカラム名
        cast_type : str
            NaNがFloatなのでint等に変換したい場合は文字列で指定する。

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data
        tmp_data = orig_data[col_name].fillna(orig_data[col_name].median())
        fill_data[col_name] = tmp_data

        if cast_type is not None:
            cls.__cast_int(fill_data, col_name, cast_type)
        return fill_data

    @classmethod
    def fill_nan_range(cls, orig_data, col_name, seed=0, cast_type=None):
        """
        NaN をすでにある値の範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
            元データ
        col_name : str
            対称のカラム名
        seed : int
            シード。指定したければどうぞ。
        cast_type : str
            NaNがFloatなのでint等に変換したい場合は文字列で指定する。

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data
        np.random.seed(seed)
        # 最大最小とその幅を取得
        data_max = orig_data[col_name].max()
        data_min = orig_data[col_name].min()
        # 指定した値の範囲でNaNを埋める関数の呼び出し
        fill_data = cls.__fill_range(fill_data, col_name, data_max, data_min, seed, cast_type)
        return fill_data

    @classmethod
    def fill_nan_user_range(cls, orig_data, col_name, data_max, data_min, seed=0, cast_type=None):
        """
        NaN を指定された値の範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
            元データ
        col_name : str
            対称のカラム名
        data_max : int or float
            乱数の最大値
        data_min : int or float
            乱数の最小値
        seed : int
            シード。指定したければどうぞ。
        cast_type : str
            NaNがFloatなのでint等に変換したい場合は文字列で指定する。

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data
        # 指定した値の範囲でNaNを埋める関数の呼び出し
        fill_data = cls.__fill_range(fill_data, col_name, data_max, data_min, seed, cast_type)
        return fill_data

    @classmethod
    def fill_nan_range_date(cls, orig_data, col_name, seed=0, date_fmt=None, change_fmt=False):
        """
        NaN をすでにある値の範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
            元データ
        col_name : str
            対称のカラム名
        seed : int
            シード。指定したければどうぞ。
        date_fmt : str
            日時データのフォーマット
        change_fmt : bool
            元の書式に戻したreturnが必要な場合True

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data

        if date_fmt is not None:
            fill_data[col_name] = pd.to_datetime(fill_data[col_name], format=date_fmt)

        # 最大最小とその幅を取得
        data_max = orig_data[col_name].max()
        data_min = orig_data[col_name].min()
        # 指定した値の範囲でNaNを埋める関数の呼び出し
        fill_data = cls.__fill_range_date(fill_data, col_name, data_max, data_min,
                                          seed=seed, date_fmt=date_fmt, change_fmt=change_fmt)
        return fill_data

    @classmethod
    def fill_nan_user_range_date(cls, orig_data, col_name, data_max, data_min, seed=0, date_fmt=None, change_fmt=False):
        """
        NaN を指定された日時の値の範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
            元データ
        col_name : str
            対称のカラム名
        data_max : Timestamp
            日時データの最大値
        data_min : Timestamp
            日時データの最小値
        seed : int
            シード。指定したければどうぞ。
        date_fmt : str
            日時データのフォーマット
        change_fmt : bool
            元の書式に戻したreturnが必要な場合True

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data

        if date_fmt is not None:
            fill_data[col_name] = pd.to_datetime(fill_data[col_name], format=date_fmt)

        # 指定した値の範囲でNaNを埋める関数の呼び出し
        fill_data = cls.__fill_range_date(fill_data, col_name, data_max, data_min,
                                          seed=seed, date_fmt=date_fmt, change_fmt=change_fmt)
        return fill_data

    @classmethod
    def fill_nan_list(cls, orig_data, col_name, from_list, weights=None, seed=0):
        """
        NaN をlistの範囲からランダムに埋める

        Parameters
        ----------
        orig_data : pandas.DataFrame
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
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        cls.__assert_all_nan(orig_data, col_name)
        fill_data = orig_data
        np.random.seed(seed)
        # 要素数を取得
        data_len = len(from_list)
        # 重みが与えられていない場合はすべて等しくする
        if weights is None:
            cls.__create_default_weights(data_len)
        assert len(weights) == data_len, \
            '[{0}] lenght({1}) is not much your input, input_len:[{2}]'.format(col_name, data_len, len(weights))
        # ランダムな値の抽出
        rand_data = np.random.choice(from_list, len(orig_data[col_name]), p=weights)

        # NaN だったところのみ乱数を格納、元データがあった部分は何もしない
        cls.__fill_nan_rand(fill_data, col_name, rand_data)
        return fill_data

    @classmethod
    def __fill_nan_rand(cls, fill_data, col_name, rand_data):
        """
        乱数で生成したデータを使ってNaNを埋める

        Parameters
        ----------
        fill_data : pandas.DataFrame
            NaNを埋めたいデータフレーム
        col_name : str
            カラム名
        rand_data : numpy.ndarray
            mp.random.*で生成したランダムデータ

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        fill_data[col_name] = fill_data[col_name].where(
            fill_data[col_name].notnull(), rand_data)

        return fill_data

    @classmethod
    def __cast_int(cls, fill_data, col_name, cast_type):
        """
        pandas.DataFrame にする際にFloatになってしまったカラムをintに変換する

        Parameters
        ----------
        fill_data : pandas.DataFrame
            NaNを埋めたいデータフレーム
        col_name : str
            カラム名
        cast_type : str
            キャストする型の文字列

        Returns
        -------
        fill_data : pandas.DataFrame
            キャスト後のデータ
        """
        fill_data[col_name] = fill_data[col_name].astype(cast_type)
        return fill_data

    @classmethod
    def __create_default_weights(cls, data_len):
        """
        重みが与えられなかった場合にすべて等しい重みlistを作成する

        Parameters
        ----------
        data_len : int
            NaNを埋めたいデータフレーム

        Returns
        -------
        weights : list
            重みlist
        """
        tmp_list = np.ones(data_len) / data_len
        weights = tmp_list.tolist()
        return weights

    @classmethod
    def __assert_all_nan(cls, orig_data, col_name):
        """
        与えられたデータがすべてNaNだった場合は警告文を出す

        Parameters
        ----------
        orig_data : pandas.DataFrame
            NaNを埋めたいデータフレーム
        col_name : str
            カラム名
        """
        assert orig_data[col_name].notnull().sum() != 0, 'all of [{0}] is null ...'.format(col_name)

    @classmethod
    def create_current_weights_names(cls, orig_data, col_name):
        """
        Parameters
        ----------
        orig_data : pandas.DataFrame
            すでにあるデータから重みを計算する
        col_name : str
            カラム名

        Returns
        -------
        weights : list
            重みlist
        """
        # すべてNaNだった場合はAssert
        cls.__assert_all_nan(orig_data, col_name)
        # 値の個数カウントと正規化
        dup_cnt = orig_data[col_name].value_counts().dropna()
        dup_cnt_std = dup_cnt / dup_cnt.sum()
        # 重みと名称リストの作成
        weights = dup_cnt_std.values.tolist()
        name_list = dup_cnt_std.index.tolist()
        return weights, name_list

    @classmethod
    def __fill_range(cls, fill_data, col_name, data_max, data_min, seed=0, cast_type=None):
        """
        NaNを与えられた最大最小範囲内の乱数で埋める

        Parameters
        ----------
        fill_data : pandas.DataFrame
            元データを格納した、これから変換予定のオブジェクト
        col_name : str
            対称のカラム名
        data_max : int or float
            乱数の最大値
        data_min : int or float
            乱数の最小値
        seed : int
            シード。指定したければどうぞ。
        cast_type : str
            NaNがFloatなのでint等に変換したい場合は文字列で指定する。

        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        np.random.seed(seed)
        data_range = data_max - data_min
        data_len = len(fill_data[col_name])

        # 乱数生成
        rand_data = data_min + np.random.rand(data_len) * data_range

        # NaN だったところのみ乱数を格納、元データがあった部分は何もしない
        cls.__fill_nan_rand(fill_data, col_name, rand_data)

        if cast_type is not None:
            cls.__cast_int(fill_data, col_name, cast_type)
        return fill_data

    @classmethod
    def __fill_range_date(cls, fill_data, col_name, data_max, data_min, seed=0, date_fmt=None, change_fmt=False):
        """
        NaNを与えられた最大最小範囲内の乱数で埋める

        Parameters
        ----------
        fill_data : pandas.DataFrame
            元データを格納した、これから変換予定のオブジェクト
        col_name : str
            対称のカラム名
        data_max : Timestamp
            乱数の最大値
        data_min : Timestamp
            乱数の最小値
        seed : int
            シード。指定したければどうぞ。
        date_fmt : str
            日時データのフォーマット
        change_fmt : bool
            date_fmtに変換する場合はTrue
        Returns
        -------
        fill_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        np.random.seed(seed)

        start_date = data_min.value // 10 ** 9
        end_date = data_max.value // 10 ** 9
        data_len = len(fill_data[col_name])

        # 乱数生成
        rand_data = pd.to_datetime(
            np.random.randint(start_date, end_date, data_len), unit='s'
        )

        # NaN だったところのみ乱数を格納、元データがあった部分は何もしない
        cls.__fill_nan_rand(fill_data, col_name, rand_data)
        # change_fmtが指定されていた場合は元の書式に変換する
        if change_fmt:
            fill_data[col_name] = fill_data[col_name].dt.strftime(date_fmt)
        return fill_data

    @classmethod
    def update_dataframe(cls, target_data, source_data, pk_target, pk_source):
        """
        データフレームのキーが一致するレコードを更新する。更新される側が大きいこと。

        Parameters
        ----------
        target_data : pandas.DataFrame
            変更されるデータフレーム. こちらのほうが大きいこと
        source_data : pandas.DataFrame
            変更するデータを持ったデータフレーム. こちらのほうが小さい
        pk_target : str
            target_data の key
        pk_source : str
            source_data の key

        Returns
        -------
        update_data : pandas.DataFrame
            NaN を埋めたデータ
        """
        def create_tmp_dataframe(df, pk):
            tmp_col = df.columns
            tmp_dataframe = pd.DataFrame(df.values, columns=tmp_col)
            tmp_dataframe.index = tmp_dataframe[pk].values
            return tmp_dataframe

        assert len(target_data) > len(source_data), 'target is smaller than source.'
        tmp_target = create_tmp_dataframe(target_data, pk_target)
        tmp_source = create_tmp_dataframe(source_data, pk_source)
        tmp_target.update(tmp_source)
        update_data = tmp_target
        return update_data
