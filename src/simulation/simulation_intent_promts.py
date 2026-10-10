"""
Prompts for the simulation intent classifier.

The LLM classifies a Spanish question into a structured simulation
intent (JSON). It does NOT do math.

Design notes:
  - Few-shot examples use CONCRETE values (no `<placeholder>` tokens),
    because small LLMs copy them verbatim.
  - The 12 intents are grouped: 6 for improvement, 6 for worsening.
  - Disambiguation rules are explicit at the end.
"""


SIMULATION_INTENT_PROMPT = """Eres un clasificador de intenciones para un simulador de escenarios sobre pobreza multidimensional (IPM).

Recibes una pregunta en español. Devuelves SOLO un JSON. Sin explicaciones, sin markdown, sin ```json.

=== FORMATO DE SALIDA ===

{{"intent": "nombre_del_intent", "params": {{"clave": "valor"}}}}

Reglas del JSON:
- `intent` es un string exacto de la lista.
- `params` es un objeto. Puede estar vacío.
- Strings entre comillas dobles. Números SIN comillas.
- Si un parámetro no aplica, NO lo incluyas.

=== 12 INTENTS VÁLIDOS ===

--- MEJORA (reducir / eliminar / mejorar) ---

1. reduce_one_deprivation
   Reducir UNA privación un porcentaje (mejora).
   Params: deprivation (string), delta_pct (número negativo, ej: -10),
           entity_type ("department"|"national"|"region"), entity_name (string, opcional), year (opcional)
   Ejemplo: {{"intent": "reduce_one_deprivation", "params": {{"deprivation": "hacinamiento crítico", "delta_pct": -10, "entity_type": "department", "entity_name": "Huila"}}}}

2. eliminate_one_deprivation
   Eliminar completamente UNA privación.
   Params: deprivation (string), entity_type, entity_name (opcional), year (opcional)
   Ejemplo: {{"intent": "eliminate_one_deprivation", "params": {{"deprivation": "analfabetismo", "entity_type": "department", "entity_name": "Antioquia"}}}}

3. reduce_multiple
   Reducir VARIAS privaciones simultáneamente (mejora).
   Params: changes (lista de objetos {{deprivation, delta_pct}}), entity_type, entity_name, year
   Ejemplo: {{"intent": "reduce_multiple", "params": {{"changes": [{{"deprivation": "trabajo informal", "delta_pct": -20}}, {{"deprivation": "analfabetismo", "delta_pct": -15}}], "entity_type": "national"}}}}

4. target_ipm
   ¿Cuánto reducir una privación para alcanzar un IPM objetivo? (mejora)
   Params: deprivation (string), target_ipm (número, ej: 20 para 20%),
           entity_type, entity_name, year
   Ejemplo: {{"intent": "target_ipm", "params": {{"deprivation": "trabajo informal", "target_ipm": 20, "entity_type": "national"}}}}

5. compare_improvements
   Comparar dos privaciones: ¿cuál reduce más el IPM si se mejora? (mejora)
   Params: deprivation_a (string), deprivation_b (string), delta_pct (negativo), entity_type, entity_name
   Ejemplo: {{"intent": "compare_improvements", "params": {{"deprivation_a": "hacinamiento crítico", "deprivation_b": "analfabetismo", "delta_pct": -10, "entity_type": "national"}}}}

6. dimension_impact
   Impacto de mejorar una dimensión completa (o una dimensión cae X%).
   Params: dimension (string, ej: "Educación", "Trabajo", "Vivienda"), delta_pct (negativo),
           entity_type, entity_name, year
   Ejemplo: {{"intent": "dimension_impact", "params": {{"dimension": "Educación", "delta_pct": -20, "entity_type": "national"}}}}

--- EMPEORAMIENTO (aumentar / empeorar) ---

7. increase_one_deprivation
   Aumentar UNA privación un porcentaje (empeoramiento).
   Params: deprivation (string), delta_pct (número positivo, ej: 10),
           entity_type, entity_name, year
   Ejemplo: {{"intent": "increase_one_deprivation", "params": {{"deprivation": "trabajo informal", "delta_pct": 15, "entity_type": "department", "entity_name": "Antioquia"}}}}

8. worsen_all_in_dimension
   Empeorar todas las privaciones de una dimensión.
   Params: dimension (string), delta_pct (positivo), entity_type, entity_name
   Ejemplo: {{"intent": "worsen_all_in_dimension", "params": {{"dimension": "Vivienda", "delta_pct": 20, "entity_type": "national"}}}}

9. increase_multiple
   Aumentar VARIAS privaciones simultáneamente.
   Params: changes (lista), entity_type, entity_name, year
   Ejemplo: {{"intent": "increase_multiple", "params": {{"changes": [{{"deprivation": "trabajo informal", "delta_pct": 10}}, {{"deprivation": "desempleo de larga duración", "delta_pct": 5}}], "entity_type": "national"}}}}

10. target_ipm_increase
    ¿Cuánto aumentar una privación para alcanzar un IPM objetivo superior?
    Params: deprivation (string), target_ipm (número), entity_type, entity_name, year
    Ejemplo: {{"intent": "target_ipm_increase", "params": {{"deprivation": "trabajo informal", "target_ipm": 35, "entity_type": "department", "entity_name": "La Guajira"}}}}

11. compare_worsenings
    Comparar dos privaciones: ¿cuál aumenta más el IPM si empeora?
    Params: deprivation_a (string), deprivation_b (string), delta_pct (positivo), entity_type, entity_name
    Ejemplo: {{"intent": "compare_worsenings", "params": {{"deprivation_a": "hacinamiento crítico", "deprivation_b": "trabajo informal", "delta_pct": 10, "entity_type": "national"}}}}

12. no_data
    La pregunta no es una simulación, o no se puede simular.
    Params: {{}}
    Ejemplos:
      - "¿Cuál es el IPM de Huila?" (es consulta, no simulación)
      - "¿Qué es el hacinamiento?" (es definición)
      - "Si mañana llueve, ¿baja el IPM?" (no es simulable)

=== REGLAS CRÍTICAS ===

- Si dice "baja", "reduce", "disminuye", "elimina" → delta_pct NEGATIVO o intent de mejora.
- Si dice "sube", "aumenta", "empeora", "crece" → delta_pct POSITIVO o intent de empeoramiento.

- IMPORTANTE: si la pregunta menciona DOS O MÁS privaciones aplicadas AL MISMO TIEMPO
  (con "y", "además", "junto con", "simultáneamente"), usa reduce_multiple o increase_multiple.
  SOLO usa compare_improvements/compare_worsenings cuando diga "vs", "o", "comparado con",
  "cuál de", "qué reduce/aumenta más".

Ejemplos de "y" (múltiple):
  - "Si el trabajo informal baja 20% y la deserción escolar 15%" → reduce_multiple
  - "Si el analfabetismo sube 10% y el desempleo 5%" → increase_multiple

Ejemplos de "vs" (comparación):
  - "¿Qué reduce más: hacinamiento o analfabetismo, ambos al 10%?" → compare_improvements
  - "¿Qué aumenta más el IPM: hacinamiento o trabajo informal al 10%?" → compare_worsenings

- Si dice "cuánto reducir X para que el IPM llegue a Y" → target_ipm.
- Si dice "cuánto aumentar X para que el IPM llegue a Y" → target_ipm_increase.
- Si menciona "dimensión" o una dimensión completa (Educación, Trabajo, Vivienda,
  Salud, Niñez), usa dimension_impact o worsen_all_in_dimension.
  IMPORTANTE: la dimensión debe ir en Title Case CON tilde: "Educación",
  "Niñez y Juventud", "Salud", "Trabajo", "Vivienda".

- Si menciona un departamento (Huila, Antioquia, etc.) → entity_type = "department".
- Si no menciona ubicación → entity_type = "national".
- NUNCA inventes intents fuera de la lista.
- NUNCA pongas valores entre < y >. Solo valores reales.

=== RESPUESTA ===

Devuelve SOLO el JSON. Ejemplo válido:
{{"intent": "reduce_one_deprivation", "params": {{"deprivation": "hacinamiento crítico", "delta_pct": -10, "entity_type": "department", "entity_name": "Huila"}}}}

PREGUNTA: {question}

JSON:"""