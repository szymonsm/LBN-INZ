class YFinanseDownloader:
    """
    Class for downloading stock info from yahoofinanse. API key(s) are NOT required.
    """

    def __init__(self, ticker_list: list[str], begin_date: str, end_date: str) -> None:
        """
        :param ticker_list: list[str], list of tickers to be downloaded
        :param begin_date: str, begin date in format YYYYMMDD
        :param end_date: str, end date in format
        """
        self.ticker_list = ticker_list
        self.begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
        self.end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        if self.end_date <= self.begin_date:
            raise ValueError("End date must be greater than begin date")
        if self.end_date > datetime.datetime.now():
            raise ValueError("End date must be less or equal than current date")

    def download_ticker(self, ticker: str, begin_date: datetime.date,
                        end_date: datetime.date) -> pd.DataFrame:
            """
            Downloads data about ticker from YahooFinanse.
            
            :param ticker: string, ticker short name
            :param begin_date: datetime.date, begin date
            :param end_date: datetime.date, end date
            :return: pd.DataFrame
            """

            df = yf.download(ticker, start=begin_date, end=end_date + datetime.timedelta(days=1))
            
            df = df.add_prefix(ticker+'_')
            df = df.reset_index()   
            if len(df)>1:
              print("Ticker data downloaded correctly")
              return df
            print("WARNING: Data not downloaded correctly, might use wrong ticker name")
            return df

    def create_df(self) -> pd.DataFrame:
          """
          Creates dataframe about tickers from YahooFinanse.
          
          :return: pd.DataFrame
          """

          df = pd.DataFrame([self.begin_date + datetime.timedelta(days=i) 
          for i in range((self.end_date - self.begin_date).days+1)],
                            columns=['Date'])
          
          for ticker in self.ticker_list:
            
            df_t = self.download_ticker(ticker, self.begin_date,self.end_date)
            
            df = pd.merge(df, df_t, on='Date', how='outer')

          df['is_weekend'] = df['Date'].apply(lambda date: date.weekday() >= 5)
          
          return df


def main():
    # This is just usage example, not part of the class
   
    #params - ^GSPC - SNP_500, EURUSD=X - EUR to USD rate, BA - Boeing Company
    ticker_list = ['^GSPC', 'EURUSD=X', 'BA'] 
    begin_date = "20230801"
    end_date = "20231029"

    yfd = YFinanseDownloader(ticker_list,begin_date,end_date)
    
    df = yfd.create_df()

