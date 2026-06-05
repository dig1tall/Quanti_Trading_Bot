from src import config
from src.data_loader import download_crypto_data
from src.check_data import validate_dataset

def main():
    print("=== Трейдинг-платформа TraidER: Сбор данных ===")

    # Запускаем универсальный загрузчик, передавая параметры из конфига
    df = download_crypto_data(
        ticker=config.TICKER, interval=config.INTERVAL, period=config.PERIOD
    )

    if not df.empty:
        print("\n[Успех] Первая фаза проекта настроена и работает автономно.")
    else:
        print("\n[Ошибка] Сбой при инициализации базы данных.")

    validate_dataset()

if __name__ == "__main__":
    main()