"""
Análisis de un comunicado nuevo (tuit real recién publicado, o hipotético) para
la sección 8.4 del TFM.

Reutiliza, literalmente, el mismo preprocesado de texto y la misma inferencia
del modelo de sentimiento que src/04_analisis_semantico.py, y la misma
función predecir_evento_importante() de src/06_modelo_predictivo.py, para que
el resultado sea comparable al del pipeline histórico.

Flujo:
1. Limpieza de texto (idéntica al módulo 2: ftfy + quitar HTML + espacios,
   luego emoji.demojize, luego normalizar menciones/URLs).
2. Sentimiento con el modelo fine-tuned (prob_negative/neutral/positive).
3. sentimiento_continuo = prob_positive - prob_negative (misma fórmula que
   src/05_analisis_impacto_mercados.py, línea 609).
4. Como es una única comunicación nueva (n=1): sentiment = sentimiento_continuo,
   intensidad_max = intensidad_media = abs(sentimiento_continuo), n_comunicaciones = 1.
5. Se coge la última fila real de dataset_modelado.csv para el ticker indicado,
   se sustituyen SOLO las 4 columnas de comunicación, y se llama a
   predecir_evento_importante() con el resto de variables financieras intactas.
6. Se compara la probabilidad antes (con los valores de comunicación reales del
   último día) y después (con los del comunicado nuevo).
"""

import re

import emoji
import ftfy
import numpy as np
import pandas as pd
import torch

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")

FEATURES_COMUNICACION = ["sentiment", "intensidad_max", "intensidad_media", "n_comunicaciones"]

ACTIVOS_CON_EVIDENCIA = ["IXIC", "XLE", "TSLA", "GSPC", "ETH-USD", "BTC-USD"]


def limpiar_texto_base(texto: str) -> str:
    """Copia literal de limpiar_texto_base() en src/04_analisis_semantico.py."""
    if pd.isna(texto):
        return ""
    texto = ftfy.fix_text(str(texto))
    texto = HTML_TAG_PATTERN.sub(" ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def preprocesar_para_transformer(texto: str) -> str:
    """Copia literal de preprocesar_para_transformer() en src/04_analisis_semantico.py."""
    texto = MENTION_PATTERN.sub("@user", texto)
    return URL_PATTERN.sub("http", texto).strip()


def preparar_texto(texto_original: str) -> str:
    """Aplica la misma cadena de limpieza que preparar_texto_modelo() del módulo 2, para un único texto."""
    texto_limpio = limpiar_texto_base(texto_original)
    texto_limpio = emoji.demojize(texto_limpio, language="es")
    return preprocesar_para_transformer(texto_limpio)


def calcular_sentimiento(texto_original: str, tokenizer, modelo) -> dict:
    """
    Calcula prob_negative/neutral/positive para un único texto con el modelo
    fine-tuned, y deriva sentimiento_continuo = prob_positive - prob_negative
    (misma fórmula que el módulo 5).
    """
    texto_modelo = preparar_texto(texto_original)

    inputs = tokenizer([texto_modelo], return_tensors="pt", truncation=True, padding=True, max_length=128)
    modelo.eval()
    with torch.no_grad():
        probs = torch.softmax(modelo(**inputs).logits, dim=-1).cpu().numpy()[0]

    prob_negative = float(probs[LABEL2ID["negative"]])
    prob_neutral = float(probs[LABEL2ID["neutral"]])
    prob_positive = float(probs[LABEL2ID["positive"]])
    sentimiento_continuo = prob_positive - prob_negative

    return {
        "texto_modelo": texto_modelo,
        "etiqueta": ID2LABEL[int(np.argmax(probs))],
        "prob_negative": prob_negative,
        "prob_neutral": prob_neutral,
        "prob_positive": prob_positive,
        "sentimiento_continuo": sentimiento_continuo,
    }


def analizar_comunicado_nuevo(texto: str, ticker: str, dataset_modelado: pd.DataFrame,
                               tokenizer, modelo, modelo_info: dict) -> dict:
    """
    Pipeline completo: sentimiento del texto nuevo -> sustitución de columnas
    de comunicación sobre la última fila real del activo -> predicción antes/después.

    Devuelve un dict con toda la información necesaria para que la capa de
    lenguaje natural (Gemini o plantilla) construya la respuesta, incluyendo
    el aviso de que esto es una lectura de sensibilidad del modelo, no una
    predicción de mercado garantizada.
    """
    if ticker not in ACTIVOS_CON_EVIDENCIA:
        raise ValueError(
            f"'{ticker}' no es uno de los activos con evidencia suficiente. "
            f"Los disponibles son: {', '.join(ACTIVOS_CON_EVIDENCIA)}."
        )

    filas_ticker = dataset_modelado[dataset_modelado["ticker"] == ticker].copy()
    if filas_ticker.empty:
        raise ValueError(f"No hay filas de '{ticker}' en dataset_modelado.csv.")

    # dataset_modelado.csv ya viene ordenado por ["ticker", "date"] al construirse
    # en construir_dataset_modelado() (src/06_modelo_predictivo.py), así que basta
    # con coger la última fila de ese activo tal cual aparece.
    if "date" in filas_ticker.columns:
        filas_ticker = filas_ticker.sort_values("date")
    ultima_fila_real = filas_ticker.iloc[-1].to_dict()

    resultado_sentimiento = calcular_sentimiento(texto, tokenizer, modelo)
    sentimiento_continuo = resultado_sentimiento["sentimiento_continuo"]

    valores_comunicacion_nuevos = {
        "sentiment": sentimiento_continuo,
        "intensidad_max": abs(sentimiento_continuo),
        "intensidad_media": abs(sentimiento_continuo),
        "n_comunicaciones": 1,
    }

    datos_antes = {k: v for k, v in ultima_fila_real.items() if k in modelo_info["feature_cols_originales"]}
    datos_despues = dict(datos_antes)
    datos_despues.update(valores_comunicacion_nuevos)

    prediccion_antes = predecir_evento_importante(datos_antes, modelo_info)
    prediccion_despues = predecir_evento_importante(datos_despues, modelo_info)

    return {
        "ticker": ticker,
        "texto_original": texto,
        "sentimiento": resultado_sentimiento,
        "valores_comunicacion_reales_ultimo_dia": {k: ultima_fila_real.get(k) for k in FEATURES_COMUNICACION},
        "valores_comunicacion_nuevos": valores_comunicacion_nuevos,
        "prediccion_antes": prediccion_antes,
        "prediccion_despues": prediccion_despues,
        "diferencia_probabilidad": round(
            prediccion_despues["probabilidad"] - prediccion_antes["probabilidad"], 4
        ),
        "aviso": (
            "Esto es una simulación de sensibilidad del modelo ante un comunicado nuevo, "
            "no una predicción real de mercado. El TFM descarta predecir la dirección del "
            "retorno (capítulo 6); esta cifra refleja únicamente cómo cambia la probabilidad "
            "de evento importante estimada por el modelo al variar las columnas de "
            "comunicación, mantiendo el resto de variables financieras del último día real."
        ),
    }


def predecir_evento_importante(datos_nuevos: dict, modelo_info: dict) -> dict:
    """Copia literal de predecir_evento_importante() en src/06_modelo_predictivo.py."""
    modelo = modelo_info["modelo"]
    columnas_esperadas = modelo_info["columnas_esperadas"]
    umbral = modelo_info["umbral_decision"]

    df_input = pd.DataFrame([datos_nuevos])
    df_input_dummies = pd.get_dummies(df_input, columns=["ticker"], prefix="ticker")
    df_input_dummies = df_input_dummies.reindex(columns=columnas_esperadas, fill_value=0)

    probabilidad = modelo.predict_proba(df_input_dummies)[0, 1]
    return {
        "probabilidad": round(float(probabilidad), 4),
        "es_evento": bool(probabilidad >= umbral),
        "umbral_usado": umbral,
    }
