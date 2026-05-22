import os
import time
import sqlite3
from datetime import datetime
import pandas as pd
import ccxt

DB_NAME = "trading_data.db"

class DataManager:
    def __init__(self, db_path=DB_NAME):
        self.db_path = db_path
        self._init_db()
        # Initialize CCXT exchange objects for public queries
        self.spot_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.futures_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def _init_db(self):
        """Initializes the SQLite database and creates the candles table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                timeframe TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                is_futures INTEGER,
                PRIMARY KEY (symbol, timeframe, timestamp, is_futures)
            )
        """)
        conn.commit()
        conn.close()

    def get_exchange(self, is_futures):
        return self.futures_exchange if is_futures else self.spot_exchange

    def download_candles(self, symbol, timeframe, start_date, end_date, is_futures):
        """
        Downloads candles directly from Binance API using pagination.
        start_date and end_date should be datetime objects.
        """
        exchange = self.get_exchange(is_futures)
        
        # Format symbol for futures if necessary (e.g. BTC/USDT:USDT in ccxt for linear futures, 
        # but let's make sure we accept standard symbols like BTC/USDT or BTC/USDT:USDT)
        query_symbol = symbol
        if is_futures and ":" not in symbol:
            query_symbol = f"{symbol}:{symbol.split('/')[-1]}"
            
        since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        all_candles = []
        
        print(f"Downloading {symbol} ({timeframe}) {'Futures' if is_futures else 'Spot'} from {start_date} to {end_date}...")
        
        while since < end_timestamp:
            try:
                # fetch_ohlcv parameters: symbol, timeframe, since, limit
                candles = exchange.fetch_ohlcv(query_symbol, timeframe, since=since, limit=1000)
                if not candles:
                    break
                
                # Filter candles that are beyond the end_timestamp
                chunk_candles = [c for c in candles if c[0] <= end_timestamp]
                if not chunk_candles:
                    break
                
                all_candles.extend(chunk_candles)
                
                # Move since forward to 1 ms after the last candle's timestamp
                last_timestamp = candles[-1][0]
                if last_timestamp <= since:
                    # Prevent infinite loop if timestamp doesn't advance
                    since += 1
                else:
                    since = last_timestamp + 1
                    
                # To be polite to the API rate limits
                time.sleep(exchange.rateLimit / 1000.0)
            except Exception as e:
                print(f"Error fetching candles: {e}")
                time.sleep(2)
                
        return all_candles

    def save_candles_to_db(self, symbol, timeframe, candles, is_futures):
        """Saves fetched candles into SQLite database using INSERT OR REPLACE."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prepare data for insertion
        is_futures_int = 1 if is_futures else 0
        insert_data = []
        for c in candles:
            # c: [timestamp, open, high, low, close, volume]
            insert_data.append((symbol, timeframe, c[0], c[1], c[2], c[3], c[4], c[5], is_futures_int))
            
        cursor.executemany("""
            INSERT OR REPLACE INTO candles 
            (symbol, timeframe, timestamp, open, high, low, close, volume, is_futures)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_data)
        
        conn.commit()
        conn.close()
        print(f"Saved {len(candles)} candles to local SQLite database.")

    def get_candles(self, symbol, timeframe, start_date_str, end_date_str, is_futures, force_download=False):
        """
        Gets candles for the specified parameters. Checks the database first.
        If data is missing for the range, downloads it, caches it, and returns the full range.
        Returns a pandas DataFrame sorted by timestamp.
        """
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        is_futures_int = 1 if is_futures else 0
        
        if not force_download:
            # Query existing range from DB
            conn = sqlite3.connect(self.db_path)
            df_existing = pd.read_sql_query("""
                SELECT * FROM candles
                WHERE symbol = ? AND timeframe = ? AND is_futures = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, conn, params=(symbol, timeframe, is_futures_int, start_ts, end_ts))
            conn.close()
            
            # Simple heuristic: if we have cached data, check if we cover the full range
            # We want to check if the database has data close to start_date and end_date.
            if not df_existing.empty:
                min_ts = df_existing['timestamp'].min()
                max_ts = df_existing['timestamp'].max()
                
                # Check timeframe in milliseconds to estimate gap tolerance (allow 2 candles gap)
                tf_ms = self._timeframe_to_ms(timeframe)
                
                start_gap = (min_ts - start_ts) <= (2 * tf_ms)
                end_gap = (end_ts - max_ts) <= (2 * tf_ms)
                
                # Also check for internal gaps (if the number of rows matches expected candles)
                expected_count = (end_ts - start_ts) // tf_ms
                actual_count = len(df_existing)
                # If we have at least 95% of expected candles, we assume we don't have massive gaps
                no_gaps = actual_count >= (expected_count * 0.95)
                
                if start_gap and end_gap and no_gaps:
                    print(f"Cache hit for {symbol} ({timeframe}) from {start_date_str} to {end_date_str}.")
                    # Convert to standard format
                    df_existing['datetime'] = pd.to_datetime(df_existing['timestamp'], unit='ms')
                    return df_existing
        
        # Otherwise, download candles
        candles = self.download_candles(symbol, timeframe, start_date, end_date, is_futures)
        if candles:
            self.save_candles_to_db(symbol, timeframe, candles, is_futures)
            
        # Retrieve full data from DB
        conn = sqlite3.connect(self.db_path)
        df_all = pd.read_sql_query("""
            SELECT * FROM candles
            WHERE symbol = ? AND timeframe = ? AND is_futures = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, conn, params=(symbol, timeframe, is_futures_int, start_ts, end_ts))
        conn.close()
        
        if not df_all.empty:
            df_all['datetime'] = pd.to_datetime(df_all['timestamp'], unit='ms')
        return df_all

    def _timeframe_to_ms(self, timeframe):
        """Converts CCXT timeframe string (e.g. '1m', '5m', '1h', '1d') to milliseconds."""
        unit = timeframe[-1]
        amount = int(timeframe[:-1])
        
        minutes = 0
        if unit == 'm':
            minutes = amount
        elif unit == 'h':
            minutes = amount * 60
        elif unit == 'd':
            minutes = amount * 60 * 24
        elif unit == 'w':
            minutes = amount * 60 * 24 * 7
        else:
            raise ValueError(f"Unknown timeframe unit: {unit}")
            
        return minutes * 60 * 1000

# Quick debug execution block
if __name__ == "__main__":
    dm = DataManager()
    # Download 2 days of BTC/USDT 1h spot data
    df = dm.get_candles("BTC/USDT", "1h", "2026-05-20", "2026-05-22", is_futures=False)
    print(df.head())
    print(f"Total candles: {len(df)}")
