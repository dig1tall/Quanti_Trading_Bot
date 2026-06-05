from src.data_loader import download_crypto_data


def main():
    print("=== Трейдинг-платформа TraidER: Сбор данных ===")

    # Здесь мы можем легко менять конфигурацию для сбора
    CONFIG = {"ticker": "BTC-USD", "interval": "1d", "period": "2y"}

    # Запускаем универсальный загрузчик
    df = download_crypto_data(
        ticker=CONFIG["ticker"],
        interval=CONFIG["interval"],
        period=CONFIG["period"],
    )

    if not df.empty:
        print("\n[Успех] Первая фаза проекта настроена и работает автономно.")
    else:
        print("\n[Ошибка] Сбой при инициализации базы данных.")


if __name__ == "__main__":
    main()