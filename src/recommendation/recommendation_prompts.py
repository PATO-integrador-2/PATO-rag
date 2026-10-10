"""
Prompts for the Recommendation module.

Contains:
    - RECOMMENDATION_INTENT_PROMPT: extracts topic, source, entity,
      dimension, focus from the user's question.
    - RECOMMENDATION_ANSWER_PROMPT: drafts concrete project proposals
      in a compact markdown table (3 or 5 rows).
"""


# ============================================================
# Intent classifier
# ============================================================
RECOMMENDATION_INTENT_PROMPT = """Eres un clasificador de intenciones para un sistema que propone proyectos y políticas públicas sobre pobreza multidimensional (IPM).

Recibes una pregunta en español. Devuelves SOLO un JSON. Sin explicaciones, sin markdown, sin ```json.

=== FORMATO DE SALIDA ===

{{"topic": "<tema principal>", "source": "<dane|cepal|both|general>", "entity": "<lugar o 'general'>", "dimension": "<dimensión IPM o 'general'>", "focus": "<enfoque libre>"}}

Reglas del JSON:
- `topic`: el tema principal de la pregunta. Ej: "hacinamiento", "educación", "empleo juvenil".
- `source`: la fuente de datos/contexto a la que se refiere la pregunta:
    * "dane" si menciona Colombia, un departamento (Antioquia, Huila, etc.), o una región colombiana (Caribe, Pacífica, etc.).
    * "cepal" si menciona un país latinoamericano (México, Chile, Ecuador, etc.), "Latinoamérica", "la región", "América Latina".
    * "both" si menciona explícitamente comparar Colombia con la región.
    * "general" si no especifica lugar.
- `entity`: si menciona un lugar (Colombia, Antioquia, México, Chile...), ponlo. Si no, "general".
- `dimension`: si el tema mapea a una dimensión del IPM, usa el valor correcto:
    * Dimensiones DANE (Colombia): "Educación", "Niñez y Juventud", "Salud", "Trabajo", "Vivienda".
    * Dimensiones CEPAL (Latinoamérica): "Vivienda", "Salud", "Educación", "Empleo y pensiones".
    * Si aplica a ambas: usa la más cercana según la fuente.
    * Si no aplica: "general".
- `focus`: una frase corta que capture lo que el usuario quiere lograr. Ej: "reducir pobreza multidimensional", "mejorar calidad educativa rural".

=== REGLAS PARA DETECTAR LA FUENTE ===

Países y regiones CEPAL (usar source="cepal"):
    Chile, Colombia, Costa Rica, Dominican Republic / República Dominicana,
    Ecuador, El Salvador, Guatemala, Honduras, Mexico / México,
    Panama / Panamá, Paraguay, Brasil, Argentina, Perú, Bolivia, Venezuela,
    Latinoamérica, América Latina, "la región", CEPAL, IPM-AL.

Departamentos y regiones DANE (usar source="dane"):
    Antioquia, Huila, Bogotá, Chocó, Nariño, Valle del Cauca, Atlántico,
    Bolívar, Santander, Córdoba, Sucre, Boyacá, Tolima, Cesar, Magdalena,
    La Guajira, Cauca, Caldas, Risaralda, Quindío, Caquetá, Putumayo,
    Meta, Casanare, Vichada, Guainía, Guaviare, Vaupés, Amazonas, Arauca,
    San Andrés, Providencia, Cundinamarca, Norte de Santander,
    Colombia, "el país", Caribe, Central, Oriental, Pacífica, Orinoquía,
    Amazonia.

=== EJEMPLOS ===

Pregunta: "¿Qué proyectos puedo implementar para reducir el hacinamiento en Antioquia?"
Respuesta: {{"topic": "hacinamiento", "source": "dane", "entity": "Antioquia", "dimension": "Vivienda", "focus": "reducir hacinamiento crítico"}}

Pregunta: "Sugiere intervenciones para mejorar la educación rural"
Respuesta: {{"topic": "educación rural", "source": "general", "entity": "general", "dimension": "Educación", "focus": "mejorar educación en zonas rurales"}}

Pregunta: "¿Cómo puedo bajar el desempleo de larga duración en Colombia?"
Respuesta: {{"topic": "desempleo de larga duración", "source": "dane", "entity": "Colombia", "dimension": "Trabajo", "focus": "reducir desempleo de larga duración"}}

Pregunta: "¿Qué proyectos para reducir la pobreza multidimensional en México?"
Respuesta: {{"topic": "pobreza multidimensional", "source": "cepal", "entity": "Mexico", "dimension": "general", "focus": "reducir pobreza multidimensional"}}

Pregunta: "Sugiere políticas para mejorar el acceso a pensiones en Chile"
Respuesta: {{"topic": "pensiones", "source": "cepal", "entity": "Chile", "dimension": "Empleo y pensiones", "focus": "mejorar acceso a pensiones"}}

Pregunta: "¿Cómo mejorar la calidad del empleo en la región?"
Respuesta: {{"topic": "calidad del empleo", "source": "cepal", "entity": "general", "dimension": "Empleo y pensiones", "focus": "mejorar calidad del empleo en Latinoamérica"}}

Pregunta: "Dame ideas de políticas para combatir la pobreza multidimensional"
Respuesta: {{"topic": "pobreza multidimensional", "source": "general", "entity": "general", "dimension": "general", "focus": "reducir pobreza multidimensional"}}

Pregunta: "¿Qué puedo hacer para reducir el analfabetismo en Colombia?"
Respuesta: {{"topic": "analfabetismo", "source": "dane", "entity": "Colombia", "dimension": "Educación", "focus": "reducir analfabetismo"}}

Pregunta: "Compara las políticas de vivienda entre Colombia y México"
Respuesta: {{"topic": "políticas de vivienda", "source": "both", "entity": "Colombia, México", "dimension": "Vivienda", "focus": "comparar políticas de vivienda entre países"}}

=== REGLA CRÍTICA ===

NUNCA inventes valores. Si algo no se menciona:
    - source → "general"
    - entity → "general"
    - dimension → "general"

=== RESPUESTA ===

Devuelve SOLO el JSON.

PREGUNTA: {question}

JSON:"""


# ============================================================
# Answer redaction (compact markdown table)
# ============================================================
RECOMMENDATION_ANSWER_PROMPT = """Eres experto en políticas públicas sobre pobreza multidimensional en Colombia y América Latina.

Propón intervenciones concretas basadas en la PREGUNTA y el CONTEXTO.

FORMATO DE RESPUESTA OBLIGATORIO:
Responde con UNA SOLA tabla markdown. NADA de párrafos introductorios, NADA de conclusión, NADA de texto fuera de la tabla.

Cantidad de proyectos:
- Si el usuario pide "más", "amplía", "5 proyectos" o similar → 5 filas.
- En cualquier otro caso → EXACTAMENTE 3 filas.

Estructura (obligatoria):
| Proyecto | Descripción | Dimensión IPM |
|---|---|---|
| Nombre1 | Descripción1 | Dimensión1 |
| Nombre2 | Descripción2 | Dimensión2 |
| Nombre3 | Descripción3 | Dimensión3 |

REGLAS:
- SOLO UNA TABLA. No la repitas. No añadas otra.
- Descripciones de MÁXIMO 15 palabras.
- Dimensiones DANE: Educación, Niñez y Juventud, Salud, Trabajo, Vivienda.
- Dimensiones CEPAL: Vivienda, Salud, Educación, Empleo y pensiones.
- NO inventes cifras ni estadísticas.
- Si el CONTEXTO no es relevante, propón igual con tu conocimiento experto.
- NO añadas nada después de la tabla.

=== CONTEXTO ===
{context}

=== PREGUNTA ===
{question}

=== RESPUESTA ===
"""