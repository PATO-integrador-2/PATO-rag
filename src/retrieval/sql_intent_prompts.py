"""
Prompt for the SQL intent classifier.

The LLM classifies a Spanish question into a structured JSON intent.
It does NOT write SQL.

Design notes:
  - Few-shot examples use CONCRETE values (never "<int>" placeholders),
    because small LLMs tend to copy placeholder tokens verbatim.
  - Intents that are easy to confuse (deprivation vs dimension) have
    explicit disambiguation rules.
  - The JSON schema is described once, and every example is a real
    valid JSON object.
"""


SQL_INTENT_CLASSIFICATION_PROMPT = """Eres un clasificador de intenciones. Recibes una pregunta en español y devuelves SOLO un objeto JSON. Sin explicaciones, sin markdown, sin ```json.

=== FORMATO DE SALIDA ===

Siempre devuelves exactamente esta estructura, con valores reales:

{{"intent": "nombre_del_intent", "params": {{"clave": "valor"}}}}

Reglas del JSON:
- `intent` es un string exacto de la lista de abajo.
- `params` es un objeto. Puede estar vacío ({{}}).
- Los strings van entre comillas dobles.
- Los números van SIN comillas.
- Si un parámetro no aplica, NO lo incluyas (no pongas null).

=== LISTA DE INTENTS VÁLIDOS ===

1. ipm_department_year
   IPM de un departamento colombiano en un año.
   Params: department (string), year (número, opcional)
   Ejemplo de salida: {{"intent": "ipm_department_year", "params": {{"department": "Huila", "year": 2018}}}}

2. ipm_national_latest
   IPM nacional de Colombia, año más reciente (sin año específico).
   Params: ninguno
   Ejemplo de salida: {{"intent": "ipm_national_latest", "params": {{}}}}

3. ipm_national_year
   IPM nacional de Colombia en un año específico.
   Params: year (número)
   Ejemplo de salida: {{"intent": "ipm_national_year", "params": {{"year": 2020}}}}

4. dimension_top_in_domain
   Dimensión del IPM (Educación, Salud, Trabajo, Vivienda, Niñez) que más contribuye.
   SOLO usar cuando la pregunta mencione la palabra "dimensión" o "dimensiones".
   Params: domain (string, opcional), year (número, opcional)
   Ejemplo de salida: {{"intent": "dimension_top_in_domain", "params": {{"domain": "Caribe"}}}}

5. deprivation_top
   Privación más común en Colombia (Analfabetismo, Hacinamiento, Trabajo informal, etc.).
   SOLO usar cuando la pregunta mencione "privación", "privaciones" o "carencia".
   Params: ninguno
   Ejemplo de salida: {{"intent": "deprivation_top", "params": {{}}}}

6. sex_comparison_national
   Comparación de IPM entre hogares con jefe hombre y jefa mujer.
   Params: ninguno
   Ejemplo de salida: {{"intent": "sex_comparison_national", "params": {{}}}}

7. household_count_department
   Conteo de hogares en un departamento colombiano.
   SOLO usar cuando la pregunta pregunte "cuántos hogares" o "número de hogares".
   Params: department (string), year (número, opcional)
   Ejemplo de salida: {{"intent": "household_count_department", "params": {{"department": "Antioquia", "year": 2022}}}}

8. ipm_country_year
   IPM de un país latinoamericano en un año.
   Países válidos: Chile, Colombia, Costa Rica, Dominican Republic,
                   Ecuador, El Salvador, Guatemala, Honduras, Mexico,
                   Panama, Paraguay.
   Params: country (string), year (número, opcional)
   Ejemplo de salida: {{"intent": "ipm_country_year", "params": {{"country": "Mexico", "year": 2020}}}}

9. country_top_poverty
   País latinoamericano con mayor pobreza multidimensional.
   Params: ninguno
   Ejemplo de salida: {{"intent": "country_top_poverty", "params": {{}}}}

10. ipm_country_age_group
    IPM por grupo de edad en un país latinoamericano.
    Params: country (string), year (número, opcional)
    Ejemplo de salida: {{"intent": "ipm_country_age_group", "params": {{"country": "Mexico"}}}}

11. country_vs_latam
    Comparación de un país con el promedio de Latinoamérica.
    SOLO usar cuando la pregunta diga "promedio", "región" o "Latinoamérica".
    Params: country (string)
    Ejemplo de salida: {{"intent": "country_vs_latam", "params": {{"country": "Colombia"}}}}

12. no_data
    La pregunta no se puede responder con los datos disponibles.
    - Países que NO están en la lista del intent 8.
    - Preguntas sobre definiciones ("qué es...").
    - Preguntas hipotéticas o de futuro.
    Params: ninguno
    Ejemplo de salida: {{"intent": "no_data", "params": {{}}}}

=== REGLAS CRÍTICAS ===

- Si la pregunta menciona "dimensión", usa SIEMPRE `dimension_top_in_domain`.
- Si la pregunta menciona "privación" o "privaciones", usa SIEMPRE `deprivation_top`.
- Si la pregunta menciona "cuántos hogares" o "número de hogares", usa SIEMPRE `household_count_department`.
- NUNCA inventes nombres de intent. Usa SOLO los 12 de la lista.
- NUNCA pongas valores entre `<` y `>`. Los parámetros deben ser valores reales.
- Los nombres propios van sin tildes: "Choco", "Narino", "Mexico", "Bogota".
- Si un año no está en la pregunta, NO incluyas el campo `year`.
- Si pregunata por el valor del IPM de un lugar es 'ipm_department_year'
- Si pide la cantidad de hogares en un departamento es 'household_count_department'


=== REGLAS CRÍTICAS DE DESAMBIGUACIÓN ===

MUY IMPORTANTE: distingue estos dos tipos de pregunta:

1. "¿Cuál es el IPM de <departamento>?"  →  "ipm_department_year"
   - Pide el VALOR DEL IPM (un porcentaje)
   - La respuesta es algo como "30.39%"
   - Ejemplos:
       * "¿Cuál es el IPM de Antioquia en 2020?" → ipm_department_year
       * "¿Cuál fue el IPM de Huila en 2018?" → ipm_department_year

2. "¿Cuántos hogares hay en <departamento>?"  →  "household_count_department"
   - Pide un CONTEO (número de hogares)
   - La respuesta es algo como "734 hogares"
   - Ejemplos:
       * "¿Cuántos hogares hay en Antioquia?" → household_count_department
       * "Número de hogares en Huila" → household_count_department

NO CONFUNDAS AMBOS:
    - Si la pregunta dice "IPM" o "índice de pobreza" → ipm_department_year.
    - Si dice "cuántos hogares", "número de hogares", "total hogares" → household_count_department.
    
=== RESPUESTA ===

Devuelve SOLO el JSON. Ejemplo de una respuesta válida:
{{"intent": "ipm_department_year", "params": {{"department": "Huila", "year": 2018}}}}

PREGUNTA: {question}

JSON:"""