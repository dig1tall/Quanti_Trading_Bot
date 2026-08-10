"""Module for live execution and market interaction with Bybit via CCXT."""

import os
import time
import logging
import ccxt
import pandas as pd
import torch

import src.config as config
from src.model import QuantiGRU
from src.features import FeatureExtractor
from src.data_preprocessing import Scaler


logger = logging.getLogger("src.execution")

class LiveExecutionEngine:
    """Handles real-time candle data fetching, feature transformation, model inference, and order execution."""

    def __init__(self):
        """Initializes CCXT exchange connection, feature extractors, and neural network checkpoint."""

        logger.info("Initializing LiveExecutionEngine...")

        ticker_raw = config.DATA_LOAD_PARAMS.get('ticker', 'BTC-USDT')
        base_symbol = ticker_raw.replace('-', '/')  # Превращаем в "BTC/USDT"

        if ':' not in base_symbol:
            self.symbol = f"{base_symbol}:USDT"
        else:
            self.symbol = base_symbol

        self.interval = config.DATA_LOAD_PARAMS.get('interval', '1d')
        logger.info(f"Working futures symbol: {self.symbol} | Timeframe: {self.interval}")

        self.api_key = config.API_PARAMS.get('api_key', '')
        self.api_secret = config.API_PARAMS.get('secret', '')
        self.enable_demo = config.API_PARAMS.get('enable_demo', True)

        if not self.api_key or not self.api_secret:
            logger.error("API keys missing from configuration or environment variables.")
            raise ValueError("API keys are not configured.")

        exchange_options = {
            'apiKey': self.api_key.strip(),
            'secret': self.api_secret.strip(),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear'  # Фьючерсы USDT
            }
        }
        self.exchange = ccxt.bybit(exchange_options)

        if self.enable_demo:
            self.exchange.enable_demo_trading(True)
            logger.info("CCXT: Successfully enabled Bybit official demo trading mode.")

        try:
            self.exchange.load_markets()
            market_info = self.exchange.market(self.symbol)
            logger.info(f"Market verified successfully. Type: {market_info.get('type')}, Exchange ID: {market_info.get('id')}")
        except Exception as e:
            logger.error(f"Failed to load markets for {self.symbol}: {e}")

        self.th_long = config.BACKTEST_PARAMS.get('threshold_long', 0.56)
        self.th_short = config.BACKTEST_PARAMS.get('threshold_short', 0.39)
        self.soft_exit_threshold = config.BACKTEST_PARAMS.get('soft_exit_threshold', 0.33)

        self.extractor = FeatureExtractor()

        self.scaler = Scaler(method=config.SCALING_PARAMS.get('method', 'robust'))
        scaler_path = os.path.join(config.PROJECT_ROOT, "models", "scaler_params.json")
        if os.path.exists(scaler_path):
            self.scaler.load(scaler_path)
            logger.info("Scaler parameters loaded for live execution mode.")
        else:
            logger.warning(f"Scaler parameters not found at {scaler_path}! Default parameters used.")

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        model_path = os.path.join(config.PROJECT_ROOT, "models", "best_quanti_model.pth")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.saved_features = checkpoint.get('feature_names', [])

            input_size = len(self.saved_features)

            self.model = QuantiGRU(
                input_size=input_size,
                hidden_size=config.MODEL_PARAMS['hidden_size'],
                num_layers=config.MODEL_PARAMS['num_layers'],
                output_size=config.MODEL_PARAMS['output_size'],
                dropout_rate=config.MODEL_PARAMS.get('dropout_rate', 0.0)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"QuantiGRU model loaded onto {self.device}. Input size: {input_size}")
        else:
            logger.error(f"Model weights file not found at path: {model_path}!")
            raise FileNotFoundError("Model checkpoint path does not exist.")

        bp = config.BACKTEST_PARAMS
        self.th_long = bp.get('threshold_long', 0.515)
        self.th_short = bp.get('threshold_short', 0.39)
        self.soft_exit_threshold = bp.get('soft_exit_threshold', 0.33)

        self.stop_loss_pct = bp.get('stop_loss', 0.105)
        self.take_profit_pct = bp.get('take_profit', 0.09)

        self.current_position = "flat"

        self.seq_len = config.MODEL_PARAMS.get('sequence_length', 30)
        self.current_position = "flat"

    def check_balance(self):
        """Queries active account balance from Unified Trading Account (UTA).

            Returns:
                Tuple[float, float]: Total balance and free balance in USDT.
        """
        try:
            balance_data = self.exchange.fetch_balance(params={'accountType': 'UNIFIED'})
            total_balance = balance_data.get('total', {}).get('USDT', 0.0)
            free_balance = balance_data.get('free', {}).get('USDT', 0.0)
            return total_balance, free_balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0, 0.0

    def get_latest_candles(self):
        """Fetches candle OHLCV data covering rolling window features and sequence length."""

        try:
            limit = 252 + self.seq_len + 10
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.interval, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df.drop(columns=['timestamp'], inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching market OHLCV candles: {e}")
            return None

    def execute_trade_decision(self, pred_class: int, probabilities: list):
        """Executes live orders or adjusts positions based on model signal probabilities.

            Args:
                pred_class: Argmax prediction index from model output logits.
                probabilities: Array of class probabilities [Short, Flat, Long].
        """
        long_score = probabilities[2]
        short_score = probabilities[0]

        logger.info(f"[Decision] Output: LONG={long_score:.4f}, SHORT={short_score:.4f} | "
                    f"Thresholds: L={self.th_long}, S={self.th_short} | "
                    f"Status: {self.current_position.upper()}")

        trade_amount = 0.01

        try:
            ticker_info = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker_info['last']
        except Exception as e:
            logger.error(f"Failed to retrieve current price: {e}")
            return

        if long_score >= self.th_long and self.current_position != "long":
            exec_amount = trade_amount * 2 if self.current_position == "short" else trade_amount

            if self.current_position == "short":
                logger.info("[TRADE] Reversing position from SHORT to LONG. Doubling order size.")

            params = {
                'positionIdx': 0
            }
            if self.stop_loss_pct and self.stop_loss_pct > 0:
                params['stopLoss'] = str(round(current_price * (1.0 - self.stop_loss_pct), 2))
            if self.take_profit_pct and self.take_profit_pct > 0:
                params['takeProfit'] = str(round(current_price * (1.0 + self.take_profit_pct), 2))

            logger.info(f"[TRADE] Submitting BUY order. Price: {current_price} | "
                        f"Volume: {exec_amount} | "
                        f"Order Params: {params}")
            try:
                self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='buy',
                    amount=exec_amount,
                    price=current_price,
                    params=params
                )
                self.current_position = "long"
            except Exception as e:
                logger.error(f"Error executing LONG order: {e}")

        elif short_score >= self.th_short and self.current_position != "short":
            exec_amount = trade_amount * 2 if self.current_position == "long" else trade_amount

            if self.current_position == "long":
                logger.info("[TRADE] Reversing position from LONG to SHORT. Doubling order size.")

            params = {
                'positionIdx': 0
            }
            if self.stop_loss_pct and self.stop_loss_pct > 0:
                params['stopLoss'] = str(round(current_price * (1.0 + self.stop_loss_pct), 2))
            if self.take_profit_pct and self.take_profit_pct > 0:
                params['takeProfit'] = str(round(current_price * (1.0 - self.take_profit_pct), 2))

            logger.info(f"[TRADE] Submitting SELL order. Price: {current_price} |"
                        f"Volume: {exec_amount} | "
                        f"Order Params: {params}")
            try:
                self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='sell',
                    amount=exec_amount,
                    price=current_price, #
                    params=params
                )
                self.current_position = "short"
            except Exception as e:
                logger.error(f"Error executing SHORT order: {e}")


        elif self.current_position == "long" and long_score < self.soft_exit_threshold:
            logger.info(f"[TRADE] Soft exit triggered for LONG (LONG score dropped to {long_score:.4f} < {self.soft_exit_threshold})")
            try:
                self.exchange.create_market_order(self.symbol, 'sell', trade_amount)
                self.current_position = "flat"
            except Exception as e:
                logger.error(f"Error closing LONG position: {e}")

        elif self.current_position == "short" and short_score < self.soft_exit_threshold:
            logger.info(f"[TRADE] Soft exit triggered for SHORT (SHORT score dropped to {short_score:.4f} < {self.soft_exit_threshold})")
            try:
                self.exchange.create_market_order(self.symbol, 'buy', trade_amount)
                self.current_position = "flat"
            except Exception as e:
                logger.error(f"Error closing SHORT position: {e}")

        else:
            logger.info("[ENGINE] Maintaining active position. No trading signals triggered.")

    def run_iteration(self):
        """Executes a full inference and trading cycle iteration."""

        logger.info("--- Starting live trading iteration ---")
        try:
            total, free = self.check_balance()
            logger.info(f"[BALANCE] Total: {total:.2f} USDT | Free: {free:.2f} USDT")

            df_candles = self.get_latest_candles()
            if df_candles is None or len(df_candles) < 252:
                logger.warning("Insufficient historical candles for feature extraction. Skipping.")
                return

            df_features = self.extractor.extract_features(df_candles)

            if self.saved_features:
                df_features = df_features[self.saved_features]

            df_scaled = self.scaler.transform(df_features)

            if len(df_scaled) < self.seq_len:
                logger.warning(f"Processed data length {len(df_scaled)} is less than required sequence length {self.seq_len}. Skipping.")
                return

            recent_pattern = df_scaled.iloc[-self.seq_len:].values
            X_tensor = torch.tensor(recent_pattern, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=-1).squeeze().tolist()
                pred_class = torch.argmax(outputs, dim=-1).item()

            self.execute_trade_decision(pred_class, probabilities)

        except Exception as e:
            logger.error(f"Critical error during iteration: {e}", exc_info=True)


def run_infinite_loop():
    """Entry point for the continuous execution loop."""

    try:
        engine = LiveExecutionEngine()
        logger.info("Quanti live execution engine running...")

        while True:
            engine.run_iteration()

            logger.info("Iteration completed. Sleeping for 1 hour...")
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Live execution loop terminated manually.")
    except Exception as e:
        logger.critical(f"Fatal error in live execution loop: {e}", exc_info=True)