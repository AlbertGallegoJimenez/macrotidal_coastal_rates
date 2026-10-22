# Preprocess beach names in the shorelines dataset.
import re
import unicodedata
import pandas as pd

def _normalizar_nombre_playa(texto):
    """Normalize case, accents, separators and repeated whitespace."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.casefold()
    texto = re.sub(r"[|;,/]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

def preparar_catalogo_playas(catalogo_limpio):
    """Build the canonical lookup once, outside the row-processing loop."""
    catalogo = {}
    for playa in pd.Series(catalogo_limpio).dropna().unique():
        nombre_original = str(playa).strip()
        nombre_normalizado = _normalizar_nombre_playa(nombre_original)
        if nombre_normalizado:
            catalogo.setdefault(nombre_normalizado, nombre_original)
    return dict(sorted(catalogo.items(), key=lambda item: len(item[0]), reverse=True))

def limpiar_playa_unica(texto_sucio, catalogo):
    """Clean one unique shoreline value using a prepared catalog."""
    if pd.isna(texto_sucio):
        return None
    texto_original = str(texto_sucio).strip()
    texto_normalizado = _normalizar_nombre_playa(texto_original)
    if texto_normalizado in catalogo:
        return catalogo[texto_normalizado]

    coincidencias = [
        nombre_original
        for nombre_normalizado, nombre_original in catalogo.items()
        if nombre_normalizado in texto_normalizado
    ]
    return ", ".join(dict.fromkeys(coincidencias)) if coincidencias else texto_original or None

def limpiar_playas(serie, catalogo_limpio):
    """Vectorized cleaner: process each unique shoreline value once."""
    catalogo = preparar_catalogo_playas(catalogo_limpio)
    valores = serie if isinstance(serie, pd.Series) else pd.Series([serie])
    mapa = {
        valor: limpiar_playa_unica(valor, catalogo)
        for valor in valores.dropna().drop_duplicates()
    }
    resultado = valores.map(mapa)
    resultado = resultado.where(valores.notna(), None)
    return resultado if isinstance(serie, pd.Series) else resultado.iloc[0]


# beach_names = profiles["Playa"].dropna().unique().tolist()
# shorelines["Playa_shoreline"] = limpiar_playas(
#     shorelines["Playa_shoreline"], beach_names
# )