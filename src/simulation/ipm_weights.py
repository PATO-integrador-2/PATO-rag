"""
Official IPM weights.

Verified against clean_data_db: the weighted sum of the 15 dep_* columns
reproduces ipm_score with max_diff = 0.0 for every household.

Sources:
    - DANE: "Algoritmo para la construcción del Índice de Pobreza
      Multidimensional – IPM", Diagram 1.
    - CEPAL: "Construcción del índice de pobreza multidimensional para
      América Latina", Diagram II.1.

Both indices use the same Alkire-Foster structure:
    IPM_household = Σ (weight_i × deprivation_i)   for i = 1..n
    A household is multidimensionally poor if IPM_household >= threshold.

DANE:
    - 5 dimensions × 0.20 each
    - 15 indicators total
    - Threshold = 5/15 ≈ 0.333
    - Unit: household

CEPAL:
    - 4 dimensions × 0.25 each
    - 12 indicators total
    - Threshold = 4/12 ≈ 0.333
    - Unit: household (reported at person level in the aggregate)
"""
from typing import Optional


# ============================================================
# DANE — Colombia
# ============================================================
# Each key is a dep_* column in vw_ipm_household_departmental.
DANE_WEIGHTS: dict[str, float] = {
    # --- Educación (0.20) ---
    "dep_low_educational_achievement":    0.10,
    "dep_illiteracy":                     0.10,
    # --- Niñez y juventud (0.20) ---
    "dep_school_lag":                     0.05,
    "dep_school_nonattendance":           0.05,
    "dep_early_childhood_care_barriers":  0.05,
    "dep_child_labor":                    0.05,
    # --- Salud (0.20) ---
    "dep_no_health_insurance":            0.10,
    "dep_health_access_barriers":         0.10,
    # --- Trabajo (0.20) ---
    "dep_long_term_unemployment":         0.10,
    "dep_informal_employment":            0.10,
    # --- Vivienda y servicios (0.20) ---
    "dep_no_improved_water_source":       0.04,
    "dep_inadequate_excreta_disposal":    0.04,
    "dep_inadequate_floor_material":      0.04,
    "dep_inadequate_wall_material":       0.04,
    "dep_critical_overcrowding":          0.04,
}

DANE_THRESHOLD = 5 / 15  # ≈ 0.3333

DANE_DIMENSIONS: dict[str, list[str]] = {
    "Educación": [
        "dep_low_educational_achievement",
        "dep_illiteracy",
    ],
    "Niñez y Juventud": [
        "dep_school_lag",
        "dep_school_nonattendance",
        "dep_early_childhood_care_barriers",
        "dep_child_labor",
    ],
    "Salud": [
        "dep_no_health_insurance",
        "dep_health_access_barriers",
    ],
    "Trabajo": [
        "dep_long_term_unemployment",
        "dep_informal_employment",
    ],
    "Vivienda": [
        "dep_no_improved_water_source",
        "dep_inadequate_excreta_disposal",
        "dep_inadequate_floor_material",
        "dep_inadequate_wall_material",
        "dep_critical_overcrowding",
    ],
}

# Spanish labels for the LLM prompt and the answer formatter.
DANE_DEPRIVATION_LABELS: dict[str, str] = {
    "dep_low_educational_achievement":    "bajo logro educativo",
    "dep_illiteracy":                     "analfabetismo",
    "dep_school_lag":                     "rezago escolar",
    "dep_school_nonattendance":           "inasistencia escolar",
    "dep_early_childhood_care_barriers":  "barreras a servicios de cuidado de la primera infancia",
    "dep_child_labor":                    "trabajo infantil",
    "dep_no_health_insurance":            "falta de aseguramiento en salud",
    "dep_health_access_barriers":         "barreras de acceso a servicios de salud",
    "dep_long_term_unemployment":         "desempleo de larga duración",
    "dep_informal_employment":            "trabajo informal",
    "dep_no_improved_water_source":       "falta de acceso a fuente de agua mejorada",
    "dep_inadequate_excreta_disposal":    "inadecuada eliminación de excretas",
    "dep_inadequate_floor_material":      "material inadecuado de pisos",
    "dep_inadequate_wall_material":       "material inadecuado de paredes",
    "dep_critical_overcrowding":          "hacinamiento crítico",
}

# Reverse lookup: label → column name
DANE_LABEL_TO_COLUMN: dict[str, str] = {
    label.lower(): col for col, label in DANE_DEPRIVATION_LABELS.items()
}


# ============================================================
# CEPAL — Latin America
# ============================================================
# Each key is a deprivation name as stored in vw_latam_ipm_deprivation_contribution.
CEPAL_WEIGHTS: dict[str, float] = {
    # --- Vivienda (0.25) ---
    "Housing conditions":    1 / 12,
    "Overcrowded":           1 / 12,
    "Internet access":       1 / 12,
    # --- Salud (0.25) ---
    "Water":                 1 / 12,
    "Sanitization":          1 / 12,
    "Health insurance":      1 / 12,
    # --- Educación (0.25) ---
    "Non-attendance or lateness": 1 / 12,
    "Educational achievement":    1 / 12,
    "Illiteracy":                 1 / 12,
    # --- Empleo y pensiones (0.25) ---
    "Labor insertion":       1 / 12,
    "Quality of employment": 1 / 12,
    "Pensions":              1 / 12,
}

CEPAL_THRESHOLD = 4 / 12  # ≈ 0.3333

CEPAL_DIMENSIONS: dict[str, list[str]] = {
    "Vivienda": ["Housing conditions", "Overcrowded", "Internet access"],
    "Salud": ["Water", "Sanitization", "Health insurance"],
    "Educación": ["Non-attendance or lateness", "Educational achievement", "Illiteracy"],
    "Empleo y pensiones": ["Labor insertion", "Quality of employment", "Pensions"],
}

CEPAL_DEPRIVATION_LABELS: dict[str, str] = {
    "Housing conditions":          "condiciones de la vivienda",
    "Overcrowded":                 "hacinamiento",
    "Internet access":             "acceso a Internet",
    "Water":                       "acceso a agua",
    "Sanitization":                "saneamiento",
    "Health insurance":            "seguro de salud",
    "Non-attendance or lateness":  "inasistencia o rezago escolar",
    "Educational achievement":     "logro educativo",
    "Illiteracy":                  "analfabetismo",
    "Labor insertion":             "inserción laboral",
    "Quality of employment":       "calidad del empleo",
    "Pensions":                    "pensiones",
}

CEPAL_LABEL_TO_COLUMN: dict[str, str] = {
    label.lower(): col for col, label in CEPAL_DEPRIVATION_LABELS.items()
}


# ============================================================
# Helper functions
# ============================================================
def get_weights(source: str) -> dict[str, float]:
    """Return the weight table for a source ('DANE' or 'CEPAL')."""
    if source.upper() == "DANE":
        return DANE_WEIGHTS
    if source.upper() == "CEPAL":
        return CEPAL_WEIGHTS
    raise ValueError(f"Fuente desconocida: {source}")


def get_threshold(source: str) -> float:
    """Return the poverty threshold for a source."""
    if source.upper() == "DANE":
        return DANE_THRESHOLD
    if source.upper() == "CEPAL":
        return CEPAL_THRESHOLD
    raise ValueError(f"Fuente desconocida: {source}")


def get_labels(source: str) -> dict[str, str]:
    """Return the column → Spanish label mapping."""
    if source.upper() == "DANE":
        return DANE_DEPRIVATION_LABELS
    if source.upper() == "CEPAL":
        return CEPAL_DEPRIVATION_LABELS
    raise ValueError(f"Fuente desconocida: {source}")


def get_dimensions(source: str) -> dict[str, list[str]]:
    """Return the dimension → columns mapping."""
    if source.upper() == "DANE":
        return DANE_DIMENSIONS
    if source.upper() == "CEPAL":
        return CEPAL_DIMENSIONS
    raise ValueError(f"Fuente desconocida: {source}")


def label_to_column(source: str, label: str) -> Optional[str]:
    """
    Map a Spanish or English deprivation label to the canonical column/name.

    For DANE: maps Spanish label → dep_* column.
    For CEPAL: maps both Spanish and English labels → canonical English name.
    """
    source = source.upper()
    label_norm = label.lower().strip()

    if source == "DANE":
        table = DANE_LABEL_TO_COLUMN
        # Reverse map: if label is already a column key (dep_*), return it
        if label_norm in DANE_DEPRIVATION_LABELS or label_norm in DANE_WEIGHTS:
            return label_norm
        # Direct match
        if label_norm in table:
            return table[label_norm]
        # Substring match
        for lab, col in table.items():
            if lab in label_norm or label_norm in lab:
                return col
        return None

    if source == "CEPAL":
        # For CEPAL, the canonical names are the English keys.
        # Accept both Spanish labels and English keys.

        # 1. Already an English key?
        for eng in CEPAL_DEPRIVATION_LABELS.keys():
            if label_norm == eng.lower():
                return eng

        # 2. Already a Spanish label?
        for eng, esp in CEPAL_DEPRIVATION_LABELS.items():
            if label_norm == esp.lower():
                return eng

        # 3. Substring match (English or Spanish)
        for eng, esp in CEPAL_DEPRIVATION_LABELS.items():
            eng_l, esp_l = eng.lower(), esp.lower()
            if label_norm in eng_l or eng_l in label_norm:
                return eng
            if label_norm in esp_l or esp_l in label_norm:
                return eng

        return None

    return None


# ============================================================
# Direct test
# ============================================================
if __name__ == "__main__":
    print("DANE weights sum:", sum(DANE_WEIGHTS.values()))  # Should be 1.0
    print("CEPAL weights sum:", sum(CEPAL_WEIGHTS.values()))  # Should be 1.0
    print("DANE threshold:", DANE_THRESHOLD)
    print("CEPAL threshold:", CEPAL_THRESHOLD)
    print("DANE dep count:", len(DANE_WEIGHTS))
    print("CEPAL dep count:", len(CEPAL_WEIGHTS))
    print()
    print("Test label_to_column:")
    print("  'hacinamiento crítico' ->", label_to_column("DANE", "hacinamiento crítico"))
    print("  'analfabetismo' ->", label_to_column("DANE", "analfabetismo"))
    print("  'internet' ->", label_to_column("CEPAL", "internet"))
    
    # ============================================================
# Extra aliases for common synonyms in user questions
# ============================================================
DANE_LABEL_ALIASES: dict[str, str] = {
    # Rezago escolar
    "desercion escolar": "dep_school_lag",
    "deserción escolar": "dep_school_lag",
    "rezago": "dep_school_lag",
    # Inasistencia escolar
    "inasistencia": "dep_school_nonattendance",
    "no asiste a la escuela": "dep_school_nonattendance",
    "no escolarizado": "dep_school_nonattendance",
    # Bajo logro
    "bajo logro": "dep_low_educational_achievement",
    "logro educativo bajo": "dep_low_educational_achievement",
    "baja educacion": "dep_low_educational_achievement",
    # Analfabetismo
    "analfabeta": "dep_illiteracy",
    # Trabajo infantil
    "trabajo de menores": "dep_child_labor",
    # Desempleo
    "desempleo": "dep_long_term_unemployment",
    "sin empleo": "dep_long_term_unemployment",
    # Trabajo informal
    "informalidad": "dep_informal_employment",
    "empleo informal": "dep_informal_employment",
    # Aseguramiento
    "sin salud": "dep_no_health_insurance",
    "sin seguro": "dep_no_health_insurance",
    # Barreras de salud
    "barreras salud": "dep_health_access_barriers",
    # Vivienda
    "sin agua": "dep_no_improved_water_source",
    "sin acueducto": "dep_no_improved_water_source",
    "sin alcantarillado": "dep_inadequate_excreta_disposal",
    "excretas": "dep_inadequate_excreta_disposal",
    "pisos de tierra": "dep_inadequate_floor_material",
    "material pisos": "dep_inadequate_floor_material",
    "paredes inadecuadas": "dep_inadequate_wall_material",
    "hacinamiento": "dep_critical_overcrowding",
    "hacinados": "dep_critical_overcrowding",
}


def label_to_column_extended(source: str, label: str) -> Optional[str]:
    """
    Extended version of label_to_column that also checks DANE_LABEL_ALIASES.
    """
    result = label_to_column(source, label)
    if result:
        return result

    if source.upper() != "DANE":
        return None

    key = label.lower().strip()
    # normalize accents
    import unicodedata
    key_norm = "".join(
        c for c in unicodedata.normalize("NFKD", key)
        if not unicodedata.combining(c)
    )

    # Direct match on aliases
    for alias, col in DANE_LABEL_ALIASES.items():
        alias_norm = "".join(
            c for c in unicodedata.normalize("NFKD", alias.lower())
            if not unicodedata.combining(c)
        )
        if alias_norm == key_norm:
            return col

    # Substring
    for alias, col in DANE_LABEL_ALIASES.items():
        alias_norm = "".join(
            c for c in unicodedata.normalize("NFKD", alias.lower())
            if not unicodedata.combining(c)
        )
        if alias_norm in key_norm or key_norm in alias_norm:
            return col

    return None