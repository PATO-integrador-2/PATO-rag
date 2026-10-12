"""
Módulo de recuperación (retriever) del RAG.

Orquesta el flujo completo de una consulta:
    1. Detectar el dominio de la pregunta (enrutador simple por palabras clave)
    2. Convertir la pregunta a embedding (embed_query)
    3. Buscar los chunks más similares en Postgres (Database.search_similar_chunks)
    4. Construir el contexto con trazabilidad (fuente, dominio, dimensión, indicador, página)
    5. Generar la respuesta con Qwen vía Ollama, forzando español
    6. Devolver la respuesta + las fuentes citadas

Uso:
    from src.retriever import Retriever
    r = Retriever()
    result = r.ask("¿Qué es hacinamiento crítico en Colombia?")
    print(result["answer"])
    for src in result["sources"]:
        print(f"- {src['document']} pág {src['page']}")
"""
import os
import re
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional

from ollama import Client
from dotenv import load_dotenv

from src.storage.database import Database
from src.ingestion.embedder import embed_query
from src.core.text_normalizer import normalize

load_dotenv()

# ============================================================
# Configuración
# ============================================================
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.7b")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")

# Enrutador: palabras clave → dominio
# Se usa para filtrar la búsqueda vectorial y evitar mezclar
# definiciones de DANE con las de CEPAL, por ejemplo.
DOMAIN_ROUTER_KEYWORDS = {
    "colombia": [
        "colombia", "colombiano", "dane", "ipm colombia",
        "departamento", "municipio colombiano",
    ],
    "cepal": [
        "cepal", "latinoamerica", "latinoamérica", "america latina",
        "región", "regional", "américa latina",
    ],
    "ophi": [
        "ophi", "oxford", "global mpi", "mpi global",
        "international", "internacional",
    ],
    "pnud": [
        "pnud", "undp", "agenda 2030", "desarrollo humano",
        "objetivos de desarrollo sostenible", "ods",
    ],
}


# ============================================================
# Retriever
# ============================================================
class Retriever:
    """Cliente del RAG para hacer consultas en lenguaje natural."""

    def __init__(self, db: Optional[Database] = None):
        """
        Inicializa el retriever. Si no se pasa una instancia de Database,
        crea una nueva.
        """
        self.db = db or Database()

    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.db:
            self.db.close()

    # --------------------------------------------------------
    # Enrutador por palabras clave
    # --------------------------------------------------------
    def detect_domain(self, question: str) -> Optional[str]:
        """
        Detecta el dominio de la pregunta por palabras clave.

        Returns:
            El nombre del dominio ('colombia', 'cepal', ...), o None si no hay match.
        """
        q_norm = normalize(question)
        for domain, keywords in DOMAIN_ROUTER_KEYWORDS.items():
            for kw in keywords:
                if normalize(kw) in q_norm:
                    return domain
        return None

    # --------------------------------------------------------
    # Recuperación de chunks
    # --------------------------------------------------------
    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        domain: Optional[str] = None,
        dimension: Optional[str] = None,
        indicator: Optional[str] = None,
    ) -> list[dict]:
        """
        Recupera los top_k chunks más similares a la pregunta.

        Args:
            question: la pregunta del usuario.
            top_k: cuántos chunks recuperar.
            domain: filtrar por dominio (si None, se autodetecta).
            dimension: filtrar por dimensión.
            indicator: filtrar por indicador.

        Returns:
            Lista de dicts con: content, page_number, domain, dimension,
            indicator, similarity, document_title, source_name.
        """
        # Autodetectar dominio si no se pasó explícitamente
        if domain is None:
            domain = self.detect_domain(question)

        # Convertir pregunta a embedding
        query_vector = embed_query(question)

        # Buscar en Postgres
        results = self.db.search_similar_chunks(
            query_embedding=query_vector,
            top_k=top_k,
            domain=domain,
            dimension=dimension,
            indicator=indicator,
        )
        return results

    # --------------------------------------------------------
    # Construcción del contexto y del prompt
    # --------------------------------------------------------
    def _build_context(self, chunks: list[dict]) -> str:
        """
        Construye el bloque de contexto con trazabilidad completa.

        Cada chunk se presenta así:
            [Fuente: DANE | Documento: X | Pág: 12 | Dominio: colombia |
            Dimensión: vivienda_servicios | Indicador: hacinamiento crítico]
            <texto del chunk>
        """
        blocks = []
        for i, c in enumerate(chunks, start=1):
            header = (
                f"--- FUENTE {i} ---\n"
                f"Institución: {c.get('source_name', 's/d')}\n"
                f"Documento: {c.get('document_title', 's/t')}\n"
                f"Página: {c.get('page_number', '?')}\n"
                f"Dominio: {c.get('domain', 's/d')}\n"
                f"Dimensión: {c.get('dimension') or 'no_clasificada'}\n"
                f"Indicador: {c.get('indicator') or 'no_clasificado'}\n"
                f"Texto:\n{c['content']}"
            )
            blocks.append(f"{header}\n{c['content']}")
        return "\n\n---\n\n".join(blocks)

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Construye el prompt final para Qwen.
        Fuerza respuesta en español, exige citar fuentes y prohíbe inventar.
        """
        return f"""Eres un experto en pobreza multidimensional en Latinoamérica.
Tu tarea es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

INSTRUCCIONES:
    1. Responde SIEMPRE en español, aunque el contexto esté en inglés.
    2. Responde ÚNICAMENTE basándote en el contenido de los campos "Texto:" de las fuentes.
    3. NO menciones la palabra "similitud", "fuente 1", "fuente 2" ni los números de 
    similitud en tu respuesta. Solo usa el contenido textual.
    4. Cita los documentos como "(Documento, pág. X)" cuando sea relevante.
    5. Si dos indicadores distintos aparecen en el contexto (por ejemplo, "analfabetismo" 
    y "bajo logro educativo"), NO los mezcles. Explica cada uno por separado.
    6. Si la información es parcial, responde con lo que haya y aclara qué falta.
    7. Si no encuentras la respuesta, di: "No tengo información suficiente en los 
    documentos para responder esa pregunta."

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA (en español, citando fuentes):"""

    # --------------------------------------------------------
    # Generación de la respuesta con Qwen
    # --------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        """Genera la respuesta con Qwen vía Ollama."""
        try:
            client = Client(host=OLLAMA_HOST)
            response = client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": 0.2,      # bajo para que sea factual
                    "top_p": 0.9,
                    "num_predict": 500,      # límite de tokens de salida
                },
            )
            return response.get("response", "").strip()
        except Exception as e:
            return f"[ERROR generando respuesta: {e}]"

    # --------------------------------------------------------
    # API pública: preguntar
    # --------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int = TOP_K,
        domain: Optional[str] = None,
        dimension: Optional[str] = None,
        indicator: Optional[str] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Ejecuta el flujo completo del RAG.

        Args:
            question: pregunta del usuario.
            top_k: cuántos chunks usar como contexto.
            domain: filtrar por dominio (si None, autodetecta).
            dimension: filtrar por dimensión.
            indicator: filtrar por indicador.
            verbose: si True, imprime pasos intermedios (útil para depurar).

        Returns:
            dict con:
                question, answer, sources (lista de dicts),
                detected_domain, chunks_used, context_length.
        """
        if not question or not question.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        # 1. Autodetectar dominio
        detected_domain = domain if domain is not None else self.detect_domain(question)

        if verbose:
            print(f"[Retriever] Dominio detectado: {detected_domain or '(sin filtro)'}")

        # 2. Recuperar chunks
        chunks = self.retrieve(
            question=question,
            top_k=top_k,
            domain=detected_domain,
            dimension=dimension,
            indicator=indicator,
        )

        if verbose:
            print(f"[Retriever] Chunks recuperados: {len(chunks)}")
            for i, c in enumerate(chunks, 1):
                print(f"  {i}. sim={c.get('similarity', 0):.3f} | "
                    f"{c.get('dimension') or '-'} / {c.get('indicator') or '-'}")

        if not chunks:
            return {
                "question": question,
                "answer": "No encontré información relevante en los documentos indexados.",
                "sources": [],
                "detected_domain": detected_domain,
                "chunks_used": 0,
                "context_length": 0,
            }

        # 3. Construir contexto y prompt
        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context)

        if verbose:
            print(f"[Retriever] Longitud del contexto: {len(context)} caracteres")
            print(f"[Retriever] Generando respuesta con {LLM_MODEL}...")

        # 4. Generar respuesta
        answer = self._generate(prompt)

        # 5. Preparar fuentes citadas (únicas)
        sources = []
        seen = set()
        for c in chunks:
            key = (c.get("source_name"), c.get("document_title"), c.get("page_number"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "source": c.get("source_name"),
                "document": c.get("document_title"),
                "page": c.get("page_number"),
                "dimension": c.get("dimension"),
                "indicator": c.get("indicator"),
                "similarity": round(c.get("similarity", 0), 4),
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "detected_domain": detected_domain,
            "chunks_used": len(chunks),
            "context_length": len(context),
        }


# ============================================================
# Prueba directa
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PRUEBA DEL RETRIEVER — RAG IPM")
    print("=" * 60)

    # Preguntas de prueba
    test_questions = [
        "¿Cual es el departamento más pobre de Colombia?",
        "¿Cómo se mide el analfabetismo en el IPM de Colombia?",
        "¿Que proyectos podría realizar para el indicador más importante del IPM para el DANE?",
    ]

    r = Retriever()
    try:
        for q in test_questions:
            print(f" PREGUNTA: {q}")


            result = r.ask(q, top_k=3, verbose=True)

            print(f"\n RESPUESTA:")
            print(result["answer"])
            print(f"\n FUENTES ({len(result['sources'])}):")
            for src in result["sources"]:
                print(f"  - {src['source']} | {src['document'][:50]} | "
                    f"pág {src['page']} | sim={src['similarity']}")
    finally:
        r.close()

    print("  FIN DE LAS PRUEBAS")
