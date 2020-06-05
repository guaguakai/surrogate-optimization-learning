import pandas as pd
import os
import datetime as dt

import quandl
import torch
from key import *
quandl.ApiConfig.api_key = API_KEY

def compute_monthly_cols(symbol_df):
    returns = symbol_df.Close.pct_change()
    # prev_12_returns = symbol_df.Close.pct_change(12) + 1
    # prev_6_returns = symbol_df.Close.pct_change(6) + 1
    # prev_3_returns = symbol_df.Close.pct_change(3) + 1

    # rolling_12 = symbol_df.Close.rolling(window=12)
    # rolling_6 = symbol_df.Close.rolling(window=6)
    # rolling_3 = symbol_df.Close.rolling(window=3)
    # rolling_2 = symbol_df.Close.rolling(window=2)

    # rolling_returns = returns.rolling(12)

    prev_365_returns = symbol_df.Close.pct_change(365)
    prev_120_returns = symbol_df.Close.pct_change(120)
    prev_30_returns = symbol_df.Close.pct_change(30)
    prev_7_returns = symbol_df.Close.pct_change(7)
    prev_3_returns = symbol_df.Close.pct_change(3)

    rolling_365 = symbol_df.Close.rolling(window=365)
    rolling_120 = symbol_df.Close.rolling(window=120)
    rolling_30 = symbol_df.Close.rolling(window=30)
    rolling_7 = symbol_df.Close.rolling(window=7)
    rolling_3 = symbol_df.Close.rolling(window=3)

    rolling_returns = returns.rolling(7)


    result_data = {
        "next10_return": returns.shift(-10),
        "next9_return": returns.shift(-9),
        "next8_return": returns.shift(-8),
        "next7_return": returns.shift(-7),
        "next6_return": returns.shift(-6),
        "next5_return": returns.shift(-5),
        "next4_return": returns.shift(-4),
        "next3_return": returns.shift(-3),
        "next2_return": returns.shift(-2),
        "next1_return": returns.shift(-1),
        "cur_return": returns,
        "prev1_return": returns.shift(1),
        "prev2_return": returns.shift(2),
        "prev3_return": returns.shift(3),
        "prev4_return": returns.shift(4),
        "prev5_return": returns.shift(5),
        "prev6_return": returns.shift(6),
        "prev7_return": returns.shift(7),
        "prev8_return": returns.shift(8),
        "prev9_return": returns.shift(9),
        "prev10_return": returns.shift(10),
        "prev_year_return": prev_365_returns,
        "prev_qtr_return": prev_120_returns,
        "prev_month_returns": prev_30_returns,
        "prev_week_returns": prev_7_returns,

        "return_rolling_mean": rolling_returns.mean(),
        "return_rolling_var": rolling_returns.var(),
        #         "return_rolling_12_min": rolling_returns.min(),
        #         "return_rolling_12_max": rolling_returns.max(),

        "rolling_365_mean": rolling_365.mean(),
        "rolling_365_var": rolling_365.var(),

        "rolling_120_mean": rolling_120.mean(),
        "rolling_120_var": rolling_120.var(),

        "rolling_30_mean": rolling_30.mean(),
        "rolling_30_var": rolling_30.var(),
        #         "rolling_12_min": rolling_12.min(),
        #         "rolling_12_max": rolling_12.max(),

        "rolling_7_mean": rolling_7.mean(),
        "rolling_7_var": rolling_7.var(),
        #         "rolling_6_min": rolling_6.min(),
        #         "rolling_6_max": rolling_6.max(),

        "rolling_3_mean": rolling_3.mean(),
        "rolling_3_var": rolling_3.var(),
        #         "rolling_3_min": rolling_3.min(),
        #         "rolling_3_max": rolling_3.max(),

        #         "rolling_2_mean": rolling_2.mean(),
        #         "rolling_2_var": rolling_2.var(),
        #         "rolling_2_min": rolling_2.min(),
        #         "rolling_2_max": rolling_2.max(),

    }
    feature_data = pd.DataFrame(result_data).dropna()
    return feature_data


def get_price_feature_matrix(price_feature_df):
    num_dates, num_assets = map(len, price_feature_df.index.levels)
    price_matrix = price_feature_df.values.reshape((num_dates, num_assets, -1))
    return price_matrix


class IndexDataLoader(object):

    def __init__(self, data_dir, index_name, start_date, end_date, collapse="monthly", overwrite=False, verbose=False):
        """
        Initialize the IndexDataLoader, the data directory structure, and any fields that are necessary down the line
        :param data_dir: Home directory to write data to, this should be specific to the index in question, i.e.
            it should already be specific to the given stock index
        :param index_name:
        :param start_date: start datetime for data collection
        :param end_date: end datetime for data collection
        :param collapse: one of {daily|weekly|monthly|quarterly|annual} interval at which to collapse
        """
        self.data_dir = data_dir
        self.index_name = index_name
        self.start_date = start_date
        self.end_date = end_date
        self.collapse = collapse
        self.overwrite = overwrite
        self.verbose = verbose

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Define data directories to write to
        self.raw_historical_price_file = \
            os.path.join(self.data_dir, "raw_historical_prices_{}_{}_{}.csv".format(start_date.date(),
                                                                                    end_date.date(),
                                                                                    collapse))

        self.raw_symbol_file = os.path.join(self.data_dir, "raw_symbols.csv")

        self.price_feature_file = os.path.join(self.data_dir,
                                               "price_feature_mat_{}_{}_{}.csv".format(start_date.date(),
                                                                                       end_date.date(),
                                                                                       collapse))
        self.torch_file = os.path.join(self.data_dir,
                                       "price_data_{}_{}_{}.pt".format(start_date.date(),
                                                                       end_date.date(),
                                                                       collapse))

    def _download_symbols(self):
        """
        Downloads symbols as a dataframe, the symbols themselves should be in the Symbols column
        :return: symbols dataframe
        """
        raise NotImplementedError

    def _download_prices(self, symbol_df):
        """
        Downloads prices given in the Symbol column of symbol_df
        Must download multilevel index with levels Date, and Symbol
        Should contain a price column called Close which will be used to get the price of the asset as well as features
        Files should be saved according to raw_price_file_format
        :return: price_df: price_dataframe from quandl
        """
        raise NotImplementedError

    def load_raw_symbols(self):
        """
        Loads symbols if they exist otherwise download them using download symbols
        :return:
        """
        if not self.overwrite and os.path.exists(self.raw_symbol_file):
            print("Loading dataset...")
            return pd.read_csv(self.raw_symbol_file)
        else:
            symbol_df = self._download_symbols()
            symbol_df.to_csv(self.raw_symbol_file, index=False)
            return symbol_df

    def get_price_feature_df(self):
        """
        Loads raw historical price data if it exists, otherwise compute the file on the fly, this adds other timeseries
        features based on rolling windows of the price
        :return:
        """
        if not self.overwrite and os.path.exists(self.price_feature_file):
            print("Loading dataset...")
            price_feature_df = pd.read_csv(self.price_feature_file, index_col=["Date", "Symbol"])
        else:

            # download prices
            if not self.overwrite and os.path.exists(self.raw_historical_price_file):
                raw_price_df = pd.read_csv(self.raw_historical_price_file, index_col=["Date", "Symbol"])
            else:
                symbol_df = self.load_raw_symbols()
                raw_price_df = self._download_prices(symbol_df)
                print("saving the data...")
                raw_price_df.to_csv(self.raw_historical_price_file)

            # filter out symbols without right number of timesteps
            max_num_timesteps = raw_price_df.groupby("Symbol").apply(lambda x: x.shape[0]).max()
            raw_price_feature_df = raw_price_df.groupby("Symbol").filter(lambda x: x.shape[0] == max_num_timesteps)

            # compute features for each symbol
            feature_df = raw_price_feature_df.groupby("Symbol", as_index=False).apply(compute_monthly_cols) \
                .droplevel(0)

            price_feature_df = feature_df.join(raw_price_df, on=["Date", "Symbol"])
            price_feature_df.index = price_feature_df.index.remove_unused_levels()
            price_feature_df.to_csv(self.price_feature_file)

        return price_feature_df

    def load_pytorch_data(self):
        """
        main function to call, this loads features and targets as torch tensors, as well as feature names, target name,
        date names, and asset names
        :return:
        """
        print(self.torch_file)
        if not self.overwrite and os.path.exists(self.torch_file):
            print("Loading pytorch data...")
            feature_mat, target_mat, feature_cols, covariance_mat, target_names, dates, symbols = torch.load(self.torch_file)
        else:
            price_feature_df = self.get_price_feature_df()
            target_names = ["next1_return"]
            covariance_names = ["next{}_return".format(i) for i in range(2,11)]
            feature_cols = [c for c in price_feature_df.columns if c not in target_names + covariance_names + ["Volume"]]
            target_mat = torch.tensor(get_price_feature_matrix(price_feature_df[target_names]))
            covariance_mat = torch.tensor(get_price_feature_matrix(price_feature_df[covariance_names]))
            feature_mat = torch.tensor(get_price_feature_matrix(price_feature_df[feature_cols]))
            dates = list(price_feature_df.index.levels[0])
            symbols = list(price_feature_df.index.levels[1])
            torch.save([feature_mat, target_mat, feature_cols, covariance_mat, target_names, dates, symbols], self.torch_file)
        return feature_mat, target_mat, feature_cols, covariance_mat, target_names, dates, symbols


class SP500DataLoader(IndexDataLoader):
    def __init__(self, data_dir="/nethome/drobinson67/aaron/diff_mip/portfolio_optimization/data/sp500",
                 index_name="sp500",
                 start_date=dt.datetime(2004, 1, 1), end_date=dt.datetime(2017, 1, 1), collapse="monthly",
                 overwrite=False, verbose=False):
        super().__init__(data_dir, index_name, start_date, end_date, collapse, overwrite, verbose)

    def _download_symbols(self):
        print("Downloading data from wiki...")
        raw_symbol_df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
        return raw_symbol_df

    def _download_prices(self, symbol_df):
        print("Downloading data from quandl...")
        # quandl.read_key(os.environ.get("QUANDL_KEY"))

        raw_tickers = symbol_df.Symbol
        tickers = "WIKI/" + raw_tickers.str.replace(".", "_")
        partial_request = list(tickers + ".11") + list(tickers + ".12")
        request_field = list(sorted(partial_request))

        print("requesting {} tickers".format(len(request_field)))

        raw_s_data = quandl.get(request_field,
                                start_date=self.start_date, end_date=self.end_date, collapse=self.collapse)
        print("processing data...")

        # only keep columns where data was found, and parse column names
        cols_to_keep = list(raw_s_data.columns[raw_s_data.columns.str.find(" - Not Found") == -1])
        raw_good_data = raw_s_data[cols_to_keep]
        raw_good_data.columns = raw_good_data.columns.str.replace("WIKI/", "")

        # split column names to form multiindex
        raw_good_data.columns = pd.MultiIndex.from_arrays(list(zip(*raw_good_data.columns.str.split(" - "))),
                                                          names=["Symbol", "Feature"])

        # get partially pivoted df with indices being Date and Symbol, columns representing the different features etc.
        price_df = pd.pivot_table(raw_good_data.reset_index().melt(id_vars=["Date"]), values="value",
                                  index=["Date", "Symbol"], columns="Feature", aggfunc="first")

        good_tickers = list(price_df.index.levels[1])
        print("found {} tickers".format(len(good_tickers)))

        price_df.rename(columns={"Adj. Close": "Close", "Adj. Volume": "Volume"}, inplace=True)

        price_df.sort_index(inplace=True)
        return price_df


class DAXDataLoader(IndexDataLoader):
    def __init__(self, data_dir="/nethome/drobinson67/aaron/diff_mip/portfolio_optimization/data/dax",
                 index_name="dax",
                 start_date=dt.datetime(2004, 1, 1), end_date=dt.datetime(2017, 1, 1), collapse="monthly",
                 overwrite=False, verbose=False):
        super().__init__(data_dir, index_name, start_date, end_date, collapse, overwrite, verbose)

    def _download_symbols(self):
        print("Downloading data from wiki...")
        raw_symbol_df = pd.read_html("https://en.wikipedia.org/wiki/DAX", header=0)[2]
        parsed_symbol_df = raw_symbol_df[["Ticker symbol"]].rename(columns={"Ticker symbol": "Symbol"})
        return parsed_symbol_df

    def _download_prices(self, symbol_df):
        print("Downloading data from quandl...")
        # quandl.read_key(os.environ.get("QUANDL_KEY"))

        raw_tickers = symbol_df.Symbol
        tickers = "FSE/" + raw_tickers + "_X"
        partial_request = list(tickers + ".4") + list(tickers + ".6")
        request_field = list(sorted(partial_request))

        print("requesting {} tickers".format(len(request_field)))

        raw_s_data = quandl.get(request_field,
                                start_date=self.start_date, end_date=self.end_date, collapse=self.collapse)

        # only keep columns where data was found, and parse column names
        cols_to_keep = list(raw_s_data.columns[raw_s_data.columns.str.find(" - Not Found") == -1])
        raw_good_data = raw_s_data[cols_to_keep]
        raw_good_data.columns = raw_good_data.columns.str.replace("FSE/|_X", "")

        # split column names to form multiindex
        raw_good_data.columns = pd.MultiIndex.from_arrays(list(zip(*raw_good_data.columns.str.split(" - "))),
                                                          names=["Symbol", "Feature"])

        # get partially pivoted df with indices being Date and Symbol, columns representing the different features etc.
        price_df = pd.pivot_table(raw_good_data.reset_index().melt(id_vars=["Date"]), values="value",
                                  index=["Date", "Symbol"], columns="Feature", aggfunc="first")

        good_tickers = list(price_df.index.levels[1])
        print("found {} tickers".format(len(good_tickers)))

        price_df.rename(columns={"Close": "Close", "Traded Volume": "Volume"}, inplace=True)

        price_df.sort_index(inplace=True)
        return price_df


if __name__ == "__main__":
    portfolio_opt_dir = os.path.abspath(os.path.dirname(__file__))
    print("portfolio_opt_dir:", portfolio_opt_dir)

    sp500_data_dir = os.path.join(portfolio_opt_dir, "data", "sp500")
    sp500_data = SP500DataLoader(sp500_data_dir, "sp500",
                                 start_date=dt.datetime(2004, 1, 1),
                                 end_date=dt.datetime(2017, 1, 1),
                                 collapse="daily",
                                 overwrite=False,
                                 verbose=True)

    # feature_mat, target_mat, feature_cols, target_name, dates, symbols = \
    sp500_data.load_pytorch_data()

    print("loaded sp500 data")

    # dax_data_dir = os.path.join(portfolio_opt_dir, "data", "dax")
    # dax_data = DAXDataLoader(dax_data_dir, "dax",
    #                          start_date=dt.datetime(2004, 1, 1),
    #                          end_date=dt.datetime(2017, 1, 1),
    #                          collapse="daily",
    #                          overwrite=False,
    #                          verbose=True)

    # # feature_mat, target_mat, feature_cols, target_name, dates, symbols = \
    # dax_data.load_pytorch_data()

    # print("loaded dax data")
