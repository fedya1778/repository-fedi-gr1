import pandas as pd
import logging

logger = logging.getLogger(__name__)


def process_features(df):

    df_processed = df.copy()
    
    cat_cols = ['Gender', 'Purchase_Category', 'BNPL_Provider', 
                'Device_Type', 'Connection_Type', 'Browser']
    
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
    
    logger.info("Признаки успешно обработаны.")
    return df_processed
