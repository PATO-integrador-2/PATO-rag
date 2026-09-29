"""
Prompts para el módulo Text-to-SQL.

Contiene las plantillas de prompt que se envían al LLM para:
    1. Generar SQL a partir de una pregunta en español.
    2. Formatear el resultado crudo como respuesta en español.
"""

# ============================================================
# Prompt para generar SQL
# ============================================================
SQL_GENERATION_PROMPT = """Eres un experto en PostgreSQL y en el Índice de Pobreza Multidimensional (IPM).

Tu tarea es traducir una pregunta en español a UNA consulta SQL válida para PostgreSQL.

REGLAS ESTRICTAS:
1. Genera SOLO la consulta SQL. Sin explicaciones, sin markdown, sin ```sql.
2. Solo se permite SELECT. Nunca INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE.
3. Usa SOLO las vistas y tablas listadas en el ESQUEMA DE ABAJO.
4. NO hagas JOINs entre vistas de grupos distintos (DANE vs CEPAL).
5. Filtra siempre por 'year' cuando la pregunta mencione un año específico.
6. Si la pregunta es ambigua, asume el año más reciente disponible.
7. Usa ILIKE para comparaciones de texto (ej: WHERE domain ILIKE '%caribe%').
8. Ordena los resultados de forma útil (por año DESC, por percentage DESC, etc.).
9. Añade LIMIT 100 por defecto si la consulta podría devolver muchas filas.
10. Si la pregunta no se puede responder con el esquema provisto, responde
    exactamente: NO_SE_PUEDE_RESPONDER

FORMATO DE RESPUESTA:
- Devuelve una sola línea con el SQL.
- Sin punto y coma al final.
- Sin comentarios.

ESQUEMA DISPONIBLE:
{schema}

PREGUNTA: {question}

SQL:"""


# ============================================================
# Prompt para formatear el resultado como respuesta
# ============================================================
ANSWER_FORMATTING_PROMPT = """Eres un experto en pobreza multidimensional en Latinoamérica.

Recibirás:
    1. La pregunta original del usuario (en español).
    2. La consulta SQL que se ejecutó.
    3. El resultado crudo de la base de datos.

Tu tarea es generar una respuesta clara, natural y en español.

REGLAS ESTRICTAS:
1. Responde SIEMPRE en español.
2. NO menciones SQL, tablas, columnas ni detalles técnicos.
3. Los porcentajes están expresados en porcentaje (0-100). Si el valor es 19.6, di "19.6%".
4. Si el resultado está vacío, di: "No encontré datos para esa consulta. Verifica el año o el lugar indicado."
5. Si el resultado tiene un solo valor, responde directamente con ese valor y su contexto.
6. Si el resultado tiene varias filas, presenta un resumen claro (tabla en markdown si es útil).
7. Si el resultado tiene muchas filas (>10), muestra las más relevantes y menciona cuántas hay en total.
8. Cita la fuente: "según datos del DANE" o "según datos de la CEPAL" según corresponda.
9. NO inventes cifras que no estén en el resultado.
10. NO hagas análisis o interpretaciones que excedan los datos.

PREGUNTA: {question}

SQL EJECUTADO: {sql}

RESULTADO (JSON):
{result}

RESPUESTA (en español):"""


# ============================================================
# Prompt para consultas "both" (DANE + CEPAL)
# ============================================================
BOTH_GROUPS_PROMPT = """Eres un experto en pobreza multidimensional en Latinoamérica.

El usuario hizo una pregunta que requiere datos de DOS fuentes distintas:
    A) DANE (Colombia) — vistas con prefijo 'vw_ipm_*'
    B) CEPAL (Latinoamérica) — vistas con prefijo 'vw_latam_ipm_*'

Tu tarea es generar DOS consultas SQL separadas, una por fuente.

REGLAS ESTRICTAS:
1. Devuelve SOLO las dos consultas, separadas EXACTAMENTE por el texto: ---SPLIT---
2. La primera consulta es para DANE, la segunda para CEPAL.
3. Cada consulta debe cumplir las mismas reglas que una consulta normal (solo SELECT, sin JOINs cruzados, etc.).
4. Si la pregunta NO se puede responder con una de las fuentes, devuelve NO_SE_PUEDE_RESPONDER para esa parte.
5. No agregues explicaciones ni markdown.

ESQUEMA DANE:
{schema_dane}

ESQUEMA CEPAL:
{schema_cepal}

PREGUNTA: {question}

SQL DANE ---SPLIT--- SQL CEPAL:"""