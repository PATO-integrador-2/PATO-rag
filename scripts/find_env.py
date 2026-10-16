import os
from pathlib import Path
from dotenv import find_dotenv, dotenv_values

print("=== cwd ===")
print(os.getcwd())

print("\n=== Archivos .env* encontrados ===")
root = Path("D:/GitHub/PATO-rag")
for p in root.rglob(".env*"):
    print(f"  {p}  ({p.stat().st_size} bytes)")

print("\n=== find_dotenv() dice ===")
print("  " + find_dotenv())

print("\n=== Valores que dotenv_values() devuelve ===")
for k, v in dotenv_values().items():
    if "PASSWORD" in k.upper():
        v = "***" if v else v
    print(f"  {k!r} = {v!r}")

print("\n=== Primeras 15 líneas del .env encontrado (repr) ===")
found = find_dotenv()
if found and os.path.exists(found):
    with open(found, "rb") as f:
        for i, line in enumerate(f.readlines()[:15], 1):
            print(f"  {i:3}: {line!r}")
else:
    print("  NO se encontró ningún .env")