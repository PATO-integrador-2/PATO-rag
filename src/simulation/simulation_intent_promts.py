"""
Prompts for the simulation intent classifier.

The LLM classifies a Spanish question into a structured simulation
intent (JSON). It does NOT do math.

Design notes:
    - Compact prompt to minimize token cost.
    - Few-shot examples with CONCRETE values.
    - Disambiguation rules kept short.
    - CEPAL countries supported via "country" entity type.
"""


SIMULATION_INTENT_PROMPT = """Clasificador de intenciones para un simulador de escenarios sobre pobreza multidimensional (IPM).

Devuelve SOLO un JSON: {{"intent": "...", "params": {{...}}}}
Sin explicaciones, sin markdown.

=== 12 INTENTS ===

MEJORA:
1. reduce_one_deprivation     → bajar UNA privación X%
2. eliminate_one_deprivation  → eliminar UNA privación
3. reduce_multiple            → bajar VARIAS privaciones ("y", "simultáneamente")
4. target_ipm                 → ¿cuánto bajar X para llegar a IPM objetivo?
5. compare_improvements       → ¿qué reduce más: A o B? ("vs", "cuál")
6. dimension_impact           → bajar toda una dimensión (Educación, Vivienda...)

EMPEORAMIENTO:
7. increase_one_deprivation   → subir UNA privación X%
8. worsen_all_in_dimension    → empeorar una dimensión entera
9. increase_multiple          → subir VARIAS privaciones
10. target_ipm_increase       → ¿cuánto subir X para llegar a IPM objetivo?
11. compare_worsenings        → ¿qué aumenta más: A o B?
12. no_data                   → no es simulación, o no se puede simular

=== PARAMS ===

Comunes:
  - delta_pct: número. Negativo=baja, Positivo=sube.
  - entity_type: "department" (departamento colombiano) | "national" (Colombia) | "country" (país CEPAL).
  - entity_name: nombre del lugar (ej. "Huila", "Mexico"). Solo si aplica.
  - year: entero opcional.

Específicos:
  - reduce_one_deprivation / increase_one_deprivation / eliminate_one_deprivation:
      deprivation (string)
  - reduce_multiple / increase_multiple:
      changes (lista de {{deprivation, delta_pct}})
  - target_ipm / target_ipm_increase:
      deprivation (string), target_ipm (número, ej. 20)
  - compare_improvements / compare_worsenings:
      deprivation_a (string), deprivation_b (string), delta_pct (número)
  - dimension_impact / worsen_all_in_dimension:
      dimension (string), delta_pct (número)

=== ENTIDADES SOPORTADAS ===

Departamentos Colombia (entity_type="department"):
    huila, antioquia, bogota, choco, narino, cauca, valle del cauca,
    atlantico, bolivar, santander, cordoba, sucre, boyaca, tolima, cesar,
    magdalena, la guajira, caldas, risaralda, quindio, caqueta, putumayo,
    meta, casanare, vichada, guainia, guaviare, vaupes, amazonas, arauca,
    san andres, providencia, cundinamarca, norte de santander.

Colombia nacional: entity_type="national", entity_name="Colombia".

Países CEPAL (entity_type="country"):
    Chile, Mexico, Costa Rica, Dominican Republic, Ecuador, El Salvador,
    Guatemala, Honduras, Panama, Paraguay.

Países NO soportados (→ "no_data"):
    Argentina, Brasil, Peru, Bolivia, Venezuela, Cuba, Nicaragua, Uruguay.

=== PRIVACIONES COMUNES ===

DANE (Colombia): hacinamiento, analfabetismo, bajo logro educativo,
    inasistencia escolar, rezago escolar, trabajo infantil, trabajo informal,
    desempleo de larga duración, sin aseguramiento en salud, barreras de salud,
    sin agua mejorada, inadecuada eliminación de excretas, pisos inadecuados,
    paredes inadecuadas.

CEPAL (países): hacinamiento, acceso a Internet, calidad del empleo,
    inserción laboral, pensiones, analfabetismo, acceso a agua, saneamiento,
    logro educativo, condiciones de la vivienda, seguro de salud,
    inasistencia o rezago.

=== REGLAS CRÍTICAS ===

1. SIGNO: "baja/reduce/elimina/mejora" → delta_pct NEGATIVO.
         "sube/aumenta/empeora/crece" → delta_pct POSITIVO.
2. UNA privación → siempre *_one_deprivation (NUNCA compare_*).
3. DOS privaciones con "y"/"simultáneamente" → *_multiple.
4. compare_* SOLO con "vs", "o", "cuál de", "qué reduce más".
5. Si NO es una simulación ("qué es", "cuál es el IPM") → "no_data".
6. País NO soportado → "no_data".
7. NUNCA inventes intents fuera de la lista.
8. NUNCA uses valores entre < y >.

=== EJEMPLOS ===

Q: "¿Qué pasaría si el hacinamiento crítico bajara 10% en Huila?"
A: {{"intent": "reduce_one_deprivation", "params": {{"deprivation": "hacinamiento", "delta_pct": -10, "entity_type": "department", "entity_name": "Huila"}}}}

Q: "¿Qué pasaría si el hacinamiento baja 10% en México?"
A: {{"intent": "reduce_one_deprivation", "params": {{"deprivation": "hacinamiento", "delta_pct": -10, "entity_type": "country", "entity_name": "Mexico"}}}}

Q: "Si el acceso a Internet sube 20% en Chile"
A: {{"intent": "increase_one_deprivation", "params": {{"deprivation": "acceso a Internet", "delta_pct": 20, "entity_type": "country", "entity_name": "Chile"}}}}

Q: "Si se elimina el analfabetismo en Antioquia"
A: {{"intent": "eliminate_one_deprivation", "params": {{"deprivation": "analfabetismo", "entity_type": "department", "entity_name": "Antioquia"}}}}

Q: "Si el trabajo informal sube 15% en Antioquia"
A: {{"intent": "increase_one_deprivation", "params": {{"deprivation": "trabajo informal", "delta_pct": 15, "entity_type": "department", "entity_name": "Antioquia"}}}}

Q: "Si el trabajo informal baja 20% y la deserción escolar 15% en Colombia"
A: {{"intent": "reduce_multiple", "params": {{"changes": [{{"deprivation": "trabajo informal", "delta_pct": -20}}, {{"deprivation": "deserción escolar", "delta_pct": -15}}], "entity_type": "national"}}}}

Q: "¿Cuánto hay que reducir el trabajo informal para que el IPM llegue a 20%?"
A: {{"intent": "target_ipm", "params": {{"deprivation": "trabajo informal", "target_ipm": 20, "entity_type": "national"}}}}

Q: "¿Qué reduce más: hacinamiento o analfabetismo?"
A: {{"intent": "compare_improvements", "params": {{"deprivation_a": "hacinamiento", "deprivation_b": "analfabetismo", "delta_pct": -10, "entity_type": "national"}}}}

Q: "Si la dimensión Educación mejora 20%"
A: {{"intent": "dimension_impact", "params": {{"dimension": "Educación", "delta_pct": -20, "entity_type": "national"}}}}

Q: "Si el hacinamiento baja 20% en Argentina"
A: {{"intent": "no_data", "params": {{}}}}

Q: "¿Qué es el hacinamiento crítico?"
A: {{"intent": "no_data", "params": {{}}}}

=== RESPUESTA ===

Devuelve SOLO el JSON.

PREGUNTA: {question}

JSON:"""