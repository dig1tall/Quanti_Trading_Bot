import os
import time
import logging
import ccxt
import pandas as pd
import torch
import numpy as np

import src.config as config
from src.model import QuantiGRU
from src.features import FeatureExtractor
from src.data_preprocessing import Scaler
from src.config import MODELS_DIR

# Настройка локального логгера модуля
logger = logging.getLogger("src.execution")

class LiveExecutionEngine:
    def __init__(self):
        logger.info("Инициализация исполнительного движка LiveExecutionEngine...")

        # 1. Настройка тикера и таймфрейма
        ticker_raw = config.DATA_LOAD_PARAMS.get('ticker', 'BTC-USDT')
        base_symbol = ticker_raw.replace('-', '/')  # Превращаем в "BTC/USDT"

        # Для фьючерсов Bybit в CCXT строго требуется формат "BTC/USDT:USDT"
        if ':' not in base_symbol:
            self.symbol = f"{base_symbol}:USDT"
        else:
            self.symbol = base_symbol

        self.interval = config.DATA_LOAD_PARAMS.get('interval', '1d')
        logger.info(f"Рабочий инструмент (Фьючерс): {self.symbol} | Таймфрейм: {self.interval}")
        self.interval = config.DATA_LOAD_PARAMS.get('interval', '1d')
        logger.info(f"Рабочий инструмент: {self.symbol} | Таймфрейм: {self.interval}")

        # 2. Извлечение API-ключей
        self.api_key = config.API_PARAMS.get('api_key', '')
        self.api_secret = config.API_PARAMS.get('secret', '')
        self.enable_demo = config.API_PARAMS.get('enable_demo', True)

        if not self.api_key or not self.api_secret:
            logger.error("[КРИТ] API-ключи отсутствуют в конфигурации или .env файле!")
            raise ValueError("API keys are not configured.")

        # 3. Инициализация CCXT Bybit
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
            logger.info("CCXT: Успешно активирован официальный режим Demo Trading Bybit.")

        # Принудительно загружаем рынки, чтобы CCXT знал спецификации контрактов
        try:
            self.exchange.load_markets()
            market_info = self.exchange.market(self.symbol)
            logger.info(f"Рынок успешно верифицирован. Тип: {market_info.get('type')}, ID на бирже: {market_info.get('id')}")
        except Exception as e:
            logger.error(f"Ошибка инициализации рынка для {self.symbol}: {e}")

        # 4. Пороги для принятия решений
        self.th_long = config.BACKTEST_PARAMS.get('threshold_long', 0.56)
        self.th_short = config.BACKTEST_PARAMS.get('threshold_short', 0.39)
        self.soft_exit_threshold = config.BACKTEST_PARAMS.get('soft_exit_threshold', 0.33)

        # 5. Инициализация компонентов Quanti
        self.extractor = FeatureExtractor()

        # Загружаем скейлер, который обучался на Train выборке
        self.scaler = Scaler(method=config.SCALING_PARAMS.get('method', 'robust'))
        scaler_path = os.path.join(config.PROJECT_ROOT, "models", "scaler_params.json")
        if os.path.exists(scaler_path):
            self.scaler.load(scaler_path)
            logger.info("Параметры Scaler успешно загружены для Live-режима.")
        else:
            logger.warning(f"Файл скейлера не найден по пути {scaler_path}! Использованы дефолтные параметры.")

        # 6. Подгрузка весов обученной нейросети
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        model_path = os.path.join(config.PROJECT_ROOT, "models", "best_quanti_model.pth")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.saved_features = checkpoint.get('feature_names', [])

            # Динамически определяем input_size на основе фичей из чекпоинта
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
            logger.info(f"Обученная модель QuantiGRU успешно загружена на {self.device}. Входной размер: {input_size}")
        else:
            logger.error(f"[КРИТ] Файл весов модели не найден по пути: {model_path}!")
            raise FileNotFoundError("Model checkpoint path does not exist.")

        bp = config.BACKTEST_PARAMS
        self.th_long = bp.get('threshold_long', 0.515)
        self.th_short = bp.get('threshold_short', 0.39)
        self.soft_exit_threshold = bp.get('soft_exit_threshold', 0.33) # порог для мягкого выхода

        # Процентные значения стопов (например, 0.02 для 2%)
        self.stop_loss_pct = bp.get('stop_loss', 0.105)
        self.take_profit_pct = bp.get('take_profit', 0.09)

        # Статус позиции (желательно в будущем запрашивать с биржи, пока оставляем flat)
        self.current_position = "flat"

        self.seq_len = config.MODEL_PARAMS.get('sequence_length', 30)
        self.current_position = "flat"  # Состояние: "flat", "long", "short"

    def check_balance(self):
        """Запрашивает актуальный баланс UTA"""
        try:
            balance_data = self.exchange.fetch_balance(params={'accountType': 'UNIFIED'})
            total_balance = balance_data.get('total', {}).get('USDT', 0.0)
            free_balance = balance_data.get('free', {}).get('USDT', 0.0)
            return total_balance, free_balance
        except Exception as e:
            logger.error(f"Ошибка при запросе баланса: {e}")
            return 0.0, 0.0

    def get_latest_candles(self):
        """Скачивает свечи с запасом под rolling-окна (252 дня) + sequence_length"""
        try:
            # Нам нужно минимум 252 строки на разгон фичей + длина последовательности GRU
            limit = 252 + self.seq_len + 10
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.interval, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df.drop(columns=['timestamp'], inplace=True)
            return df
        except Exception as e:
            logger.error(f"Ошибка получения свечей с биржи: {e}")
            return None

    def execute_trade_decision(self, pred_class: int, probabilities: list):
        """Управляет позициями на основе выхлопа нейросети и риск-менеджмента"""
        long_score = probabilities[2]
        short_score = probabilities[0]

        logger.info(f"[Решение] Модель выдала: LONG={long_score:.4f}, SHORT={short_score:.4f} | Пороги: L={self.th_long}, S={self.th_short} | Статус: {self.current_position.upper()}")

        # Минимальный тестовый лот
        trade_amount = 0.01

        try:
            # Получаем текущую цену для расчета уровней TP/SL ордеров
            ticker_info = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker_info['last']
        except Exception as e:
            logger.error(f"Не удалось получить текущую цену с биржи: {e}")
            return

        if long_score >= self.th_long and self.current_position != "long":
            exec_amount = trade_amount * 2 if self.current_position == "short" else trade_amount

            if self.current_position == "short":
                logger.info("[TRADE] Обнаружен переворот из ШОРТА в ЛОНГ. Объем удвоен.")

            params = {
                'positionIdx': 0  # Явно указываем One-Way режим для фьючерсов
            }
            if self.stop_loss_pct and self.stop_loss_pct > 0:
                params['stopLoss'] = str(round(current_price * (1.0 - self.stop_loss_pct), 2))
            if self.take_profit_pct and self.take_profit_pct > 0:
                params['takeProfit'] = str(round(current_price * (1.0 + self.take_profit_pct), 2))

            logger.info(f"[TRADE] Отправка ордера BUY. Цена: {current_price} | Итоговый объем: {exec_amount} | TP/SL параметры: {params}")
            try:
                # Использование явного create_order решает проблему с decimal.ConversionSyntax
                self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='buy',
                    amount=exec_amount,
                    price=current_price, # Передача текущей цены страхует от бага с None внутри CCXT
                    params=params
                )
                self.current_position = "long"
            except Exception as e:
                logger.error(f"Ошибка исполнения LONG: {e}")

        elif short_score >= self.th_short and self.current_position != "short":
            exec_amount = trade_amount * 2 if self.current_position == "long" else trade_amount

            if self.current_position == "long":
                logger.info("[TRADE] Обнаружен переворот из ЛОНГА в ШОРТ. Объем удвоен.")

            params = {
                'positionIdx': 0  # Явно указываем One-Way режим для фьючерсов
            }
            if self.stop_loss_pct and self.stop_loss_pct > 0:
                params['stopLoss'] = str(round(current_price * (1.0 + self.stop_loss_pct), 2))
            if self.take_profit_pct and self.take_profit_pct > 0:
                params['takeProfit'] = str(round(current_price * (1.0 - self.take_profit_pct), 2))

            logger.info(f"[TRADE] Отправка ордера SELL. Цена: {current_price} | Итоговый объем: {exec_amount} | TP/SL параметры: {params}")
            try:
                # Использование явного create_order решает проблему с decimal.ConversionSyntax
                self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='sell',
                    amount=exec_amount,
                    price=current_price, # Передача текущей цены страхует от бага с None внутри CCXT
                    params=params
                )
                self.current_position = "short"
            except Exception as e:
                logger.error(f"Ошибка исполнения SHORT: {e}")

        # --- ЛОГИКА МЯГКОГО ВЫХОДА В КЭШ (ИЗ БЭКТЕСТА) ---
        elif self.current_position == "long" and long_score < self.soft_exit_threshold:
            logger.info(f"[TRADE] Мягкий выход из ЛОНГА (скор LONG упал до {long_score:.4f} < {self.soft_exit_threshold})")
            try:
                self.exchange.create_market_order(self.symbol, 'sell', trade_amount)
                self.current_position = "flat"
            except Exception as e:
                logger.error(f"Ошибка закрытия LONG: {e}")

        elif self.current_position == "short" and short_score < self.soft_exit_threshold:
            logger.info(f"[TRADE] Мягкий выход из ШОРТА (скор SHORT упал до {short_score:.4f} < {self.soft_exit_threshold})")
            try:
                self.exchange.create_market_order(self.symbol, 'buy', trade_amount)
                self.current_position = "flat"
            except Exception as e:
                logger.error(f"Ошибка закрытия SHORT: {e}")

        else:
            logger.info("[ДВИЖОК] Удерживаем текущую позицию / Сигналов нет.")

    def run_iteration(self):
        """Полный цикл итерации лайв-трейдинга"""
        logger.info("--- Начало итерации торгового цикла ---")
        try:
            # 1. Проверяем баланс для логов
            total, free = self.check_balance()
            logger.info(f"[БАЛАНС] Всего: {total:.2f} USDT | Свободно: {free:.2f} USDT")

            # 2. Скачиваем свежую историю свечей
            df_candles = self.get_latest_candles()
            if df_candles is None or len(df_candles) < 252:
                logger.warning("Недостаточно данных для генерации признаков. Пропуск.")
                return

            # 3. Наш Feature Engineering
            df_features = self.extractor.extract_features(df_candles)

            # Убеждаемся, что колонки идут строго в том порядке, в котором их ждет модель
            if self.saved_features:
                df_features = df_features[self.saved_features]

            # 4. Масштабирование (Скейлинг) признаков
            df_scaled = self.scaler.transform(df_features)

            # 5. Сборка 3D-тензора под GRU [Batch=1, Seq_Len, Features]
            if len(df_scaled) < self.seq_len:
                logger.warning(f"После очистки осталось {len(df_scaled)} строк, а нужно {self.seq_len}. Пропуск.")
                return

            # Берем последние N дней для формирования текущего паттерна
            recent_pattern = df_scaled.iloc[-self.seq_len:].values
            X_tensor = torch.tensor(recent_pattern, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 6. Инференс нейросети
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=-1).squeeze().tolist()
                pred_class = torch.argmax(outputs, dim=-1).item()

            # 7. Исполнение торгового приказа
            self.execute_trade_decision(pred_class, probabilities)

        except Exception as e:
            logger.error(f"Критическая ошибка в итерации: {e}", exc_info=True)


def run_infinite_loop():
    """Точка входа бесконечного цикла, вызываемая из main.py"""
    try:
        engine = LiveExecutionEngine()
        logger.info("Робот Quanti запущен в бесконечном цикле.")

        while True:
            engine.run_iteration()

            # Так как таймфрейм дневной (1d), опрашивать рынок каждую секунду нет смысла.
            # Опрос раз в 1 час (3600 сек) — идеальный баланс, чтобы не спамить биржу.
            logger.info("Итерация завершена. Засыпаем на 1 час...")
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info("[STOP] Робот остановлен пользователем вручную.")
    except Exception as e:
        logger.critical(f"[КРАШ] Сбой бесконечного цикла: {e}", exc_info=True)