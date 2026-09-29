"""
Clasificador de intención de preguntas.

Determina qué módulo debe responder la pregunta:
    - 'simulation'     → módulo de simulación de escenarios
    - 'data'           → módulo Text-to-SQL
    - 'definition'     → módulo RAG clásico (definiciones, metodología)
    - 'recommendation' → (futuro) módulo de propuestas de proyectos

Reglas:
    - Se basa en patrones textuales (palabras clave).
    - Prioridad: simulation > data > definition.
    - Fallback por defecto: 'definition'.
"""
import re
from typing import Literal

from src.core.text_normalizer import normalize


Intent = Literal["simulation", "data", "definition", "recommendation"]


# ============================================================
# Patrones por categoría
# ============================================================
SIMULATION_PATTERNS = [
    # Hipotéticas explícitas
    r"\bque pasaria si\b",
    r"\bqué pasaría si\b",
    r"\bsi se redujera\b",
    r"\bsi se aumentara\b",
    r"\bsi bajara\b",
    r"\bsi subiera\b",
    r"\bsi se eliminara\b",
    r"\bsi se duplicara\b",
    r"\bsi disminuyera\b",
    r"\bsi creciera\b",
    # Impacto / efecto
    r"\bcual seria el impacto\b",
    r"\bcuál sería el impacto\b",
    r"\bimpacto de reducir\b",
    r"\bimpacto de aumentar\b",
    r"\bimpacto de eliminar\b",
    r"\bque efecto tendria\b",
    r"\bqué efecto tendría\b",
    # Proyección / estimación
    r"\bescenario hipotetico\b",
    r"\bescenario hipotético\b",
    r"\bsimula\b",
    r"\bproyecta\b",
    r"\bestima el efecto\b",
    r"\bcomo cambiaria\b",
    r"\bcómo cambiaría\b",
]

DATA_PATTERNS = [
    # Interrogativos de valor
    r"\bcuanto\b",
    r"\bcuánto\b",
    r"\bcuantos\b",
    r"\bcuántos\b",
    r"\bcual fue\b",
    r"\bcuál fue\b",
    r"\bcual es el valor\b",
    r"\bcuál es el valor\b",
    r"\bdame el dato\b",
    r"\bmuestrame el dato\b",
    r"\bmuéstrame el dato\b",
    # Cantidad / magnitud
    r"\bnumero de\b",
    r"\bnúmero de\b",
    r"\bporcentaje de\b",
    r"\btotal de\b",
    r"\bpromedio de\b",
    r"\bcifra de\b",
    r"\bvalor de\b",
    # Comparaciones
    r"\bcompara\b",
    r"\bcomparacion\b",
    r"\bcomparación\b",
    r"\bdiferencia entre\b",
    # Años explícitos
    r"\ben el ano\b",
    r"\ben el año\b",
    r"\ben 19\d{2}\b",
    r"\ben 20\d{2}\b",
    r"\bdel ano\b",
    r"\bdel año\b",
    # Rankings (ampliado)
    r"\branking\b",
    r"\bel mas alto\b",
    r"\bel más alto\b",
    r"\bel mas bajo\b",
    r"\bel más bajo\b",
    r"\bcon mayor\b",        
    r"\bcon menor\b",       
    r"\bmas alto\b",         
    r"\bmas bajo\b",         
    r"\bmás alto\b",
    r"\bmás bajo\b",
    r"\bmayor ipm\b",       
    r"\bmenor ipm\b",        
    r"\bmayor pobreza\b",    
    r"\bmenor pobreza\b",   
    r"\btop \d+\b",
]

RECOMMENDATION_PATTERNS = [
    # Proyectos e intervenciones
    r"\bque proyectos\b",
    r"\bqué proyectos\b",
    r"\bpropone\b",
    r"\bpropón\b",
    r"\bpropon\b",
    r"\bpropuesta de proyecto\b",
    r"\bideas para\b",
    r"\bintervencion\b",
    r"\bintervención\b",
    r"\bintervenciones\b",
    # Reducir / mejorar (ampliado)
    r"\bcomo reducir\b",
    r"\bcómo reducir\b",
    r"\bcomo se puede reducir\b",        
    r"\bcómo se puede reducir\b",         
    r"\bcomo puedo reducir\b",            
    r"\bcómo puedo reducir\b",            
    r"\bcomo podemos reducir\b",         
    r"\bcomo mejorar\b",
    r"\bcómo mejorar\b",
    r"\bcomo puedo mejorar\b",           
    r"\bcómo puedo mejorar\b",           
    r"\bcomo se puede mejorar\b",       
    r"\bcomo solucionar\b",              
    r"\bcómo solucionar\b",               
    r"\bque se puede hacer para\b",       
    r"\bqué se puede hacer para\b",       
    # Qué puedo hacer / implementar
    r"\bque puedo hacer\b",
    r"\bqué puedo hacer\b",
    r"\bque se puede implementar\b",
    r"\bqué se puede implementar\b",
    # Sugerencias
    r"\bsugiere\b",
    r"\bsugiéreme\b",
    r"\bsugiereme\b",
    r"\bdame ideas\b",                    
]


# ============================================================
# Clasificador
# ============================================================
def _matches_any(text: str, patterns: list[str]) -> bool:
    """Devuelve True si el texto matchea al menos uno de los patrones."""
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def classify_intent(question: str) -> Intent:
    """
    Clasifica la pregunta en uno de los 4 tipos.
    Prioridad: recommendation > simulation > data > definition.
    """
    if not question or not question.strip():
        return "definition"

    q_norm = normalize(question)

    if _matches_any(q_norm, RECOMMENDATION_PATTERNS):
        return "recommendation"
    if _matches_any(q_norm, SIMULATION_PATTERNS):
        return "simulation"
    if _matches_any(q_norm, DATA_PATTERNS):
        return "data"
    return "definition"


def classify_intent_with_score(question: str) -> dict:
    """
    Versión extendida que devuelve también el score de cada categoría.
    Útil para depurar el clasificador.
    """
    q_norm = normalize(question)

    scores = {
        "recommendation": sum(
            1 for p in RECOMMENDATION_PATTERNS
            if re.search(p, q_norm, flags=re.IGNORECASE)
        ),
        "simulation": sum(
            1 for p in SIMULATION_PATTERNS
            if re.search(p, q_norm, flags=re.IGNORECASE)
        ),
        "data": sum(
            1 for p in DATA_PATTERNS
            if re.search(p, q_norm, flags=re.IGNORECASE)
        ),
    }

    # Definición es fallback
    max_score = max(scores.values())
    if max_score == 0:
        winner = "definition"
    else:
        # Orden de prioridad en caso de empate
        priority = ["recommendation", "simulation", "data"]
        winner = next(k for k in priority if scores[k] == max_score)

    return {
        "question": question,
        "intent": winner,
        "scores": scores,
    }


# ============================================================
# Prueba directa
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  PRUEBA DEL CLASIFICADOR DE INTENCIÓN")
    print("=" * 70)

    test_cases = [
        # --- definition ---
        ("¿Qué es el hacinamiento crítico?", "definition"),
        ("¿Cómo se mide el analfabetismo?", "definition"),
        ("¿Cuáles son las dimensiones del IPM?", "definition"),
        ("Explícame el concepto de pobreza multidimensional", "definition"),
        ("Dame la definición de trabajo informal", "definition"),

        # --- data ---
        ("¿Cuál fue el IPM de Huila en 2018?", "data"),
        ("¿Cuántos hogares hay en Antioquia?", "data"),
        ("¿Cuál es el promedio de IPM en la región Caribe?", "data"),
        ("Compara el IPM de Bogotá y Antioquia en 2022", "data"),
        ("Dame el porcentaje de pobreza en 2023", "data"),
        ("¿Cuál es el departamento con mayor IPM?", "data"),
        ("Top 5 de departamentos con menor pobreza", "data"),

        # --- simulation ---
        ("¿Qué pasaría si el hacinamiento bajara 10% en Antioquia?", "simulation"),
        ("Si se eliminara la privación de agua mejorada, ¿cuánto bajaría el IPM?", "simulation"),
        ("¿Cuál sería el impacto de reducir el desempleo a la mitad?", "simulation"),
        ("Estima el efecto de aumentar la escolaridad en 2 años", "simulation"),
        ("Simula un escenario donde se duplica la inversión social", "simulation"),

        # --- recommendation ---
        ("¿Qué proyectos puedo implementar para reducir el IPM en Antioquia?", "recommendation"),
        ("Sugiere ideas para mejorar la educación en zonas rurales", "recommendation"),
        ("Propón intervenciones para el hacinamiento crítico", "recommendation"),
        ("¿Cómo puedo reducir la pobreza multidimensional?", "recommendation"),
    ]

    correct = 0
    total = len(test_cases)

    for question, expected in test_cases:
        result = classify_intent_with_score(question)
        got = result["intent"]
        status = "✅" if got == expected else "❌"
        if got == expected:
            correct += 1
        print(f"\n{status} PREGUNTA: {question}")
        print(f"   Esperado: {expected:<15} | Obtenido: {got}")
        print(f"   Scores: {result['scores']}")

    print(f"\n{'=' * 70}")
    print(f"  RESULTADO: {correct}/{total} correctas ({100*correct/total:.1f}%)")
    print(f"{'=' * 70}")