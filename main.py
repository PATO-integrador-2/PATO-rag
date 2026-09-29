# main.py
"""
Punto de entrada del proyecto RAG.
Ejecuta: python main.py
"""
import sys
import os

# Asegura que la raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG - IPM Latinoamérica")
    print("=" * 60)
    print("\nMódulos disponibles:")
    print("  python -m src.text_normalizer    → prueba de normalización")
    print("  python -m src.taxonomy_loader    → prueba de carga de taxonomía")
    print("  python -m src.pdf_extractor      → prueba de extracción de PDFs")
    print("  python -m src.chunker            → prueba de chunking y clasificación")
    print("\nEjecuta cualquiera de ellos desde la raíz del proyecto.")