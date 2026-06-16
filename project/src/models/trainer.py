import os
import joblib
import logging
from lightgbm import LGBMClassifier
from src.data.loader import load_data
from src.features.preprocessor import process_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def train_and_save_model(data_path: str, save_path: str = 'artifacts/models/production_model.pkl'):
    
    logger.info(f"Загрузка очищенных данных для обучения из {data_path}...")
    df = load_data(data_path)

    X = df.drop(columns=['Repayment_Status'])
    y = df['Repayment_Status']
    
    X_processed = process_features(X)

    logger.info("Инициализация наилучшей конфигурации LightGBM из ноутбука 02...")
    
    model = LGBMClassifier(
        n_estimators=150,          # Оптимальное количество деревьев из экспериментов
        learning_rate=0.05,        # Стабильный шаг обучения для защиты от переобучения
        max_depth=6,               # Ограничение глубины дерева
        random_state=42,           # Фиксация сида для воспроизводимости
        n_jobs=-1,                 # Использование всех ядер процессора
        # scale_pos_weight=3.0     # Раскомментируй и укажи коэффициент, если с ним Recall был выше
    )

    logger.info(f"Запуск обучения модели на {len(X_processed)} строках данных...")
    model.fit(X_processed, y)
    logger.info("Обучение успешно завершено!")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    joblib.dump(model, save_path)
    logger.info(f"Наилучшая Production-модель успешно сохранена по пути: {save_path}")

if __name__ == "__main__":
    DATA_PATH = 'data/cleaned_bnpl_data.csv'
    train_and_save_model(DATA_PATH)