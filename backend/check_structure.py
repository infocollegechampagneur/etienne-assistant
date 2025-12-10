#!/usr/bin/env python3
"""Script de diagnostic pour vérifier la structure sur Render"""
import os
import sys
from pathlib import Path

print("=" * 60)
print("DIAGNOSTIC STRUCTURE RENDER")
print("=" * 60)

# 1. Working directory actuel
cwd = Path.cwd()
print(f"\n1. Working Directory: {cwd}")

# 2. Localisation de ce script
script_path = Path(__file__).resolve()
print(f"2. Script Path: {script_path}")
print(f"   Script Parent: {script_path.parent}")

# 3. Vérifier dossier utils relatif au script
utils_relative = script_path.parent / "utils"
print(f"\n3. Utils (relatif au script): {utils_relative}")
print(f"   Existe? {utils_relative.exists()}")

if utils_relative.exists():
    try:
        contents = list(utils_relative.iterdir())
        print(f"   Contenu: {[f.name for f in contents]}")
    except Exception as e:
        print(f"   Erreur listage: {e}")

# 4. Vérifier dossier utils relatif au CWD
utils_cwd = cwd / "utils"
print(f"\n4. Utils (relatif au CWD): {utils_cwd}")
print(f"   Existe? {utils_cwd.exists()}")

if utils_cwd.exists():
    try:
        contents = list(utils_cwd.iterdir())
        print(f"   Contenu: {[f.name for f in contents]}")
    except Exception as e:
        print(f"   Erreur listage: {e}")

# 5. Chercher utils dans les parents
print(f"\n5. Recherche dans les parents:")
current = cwd
for i in range(5):
    utils_check = current / "utils"
    print(f"   {current}/utils: {utils_check.exists()}")
    if utils_check.exists():
        try:
            contents = list(utils_check.iterdir())
            print(f"      → Contenu: {[f.name for f in contents]}")
        except:
            pass
    current = current.parent
    if current == current.parent:  # Racine atteinte
        break

# 6. Python path
print(f"\n6. Python sys.path:")
for p in sys.path[:10]:
    print(f"   - {p}")

# 7. Lister le contenu du répertoire backend
backend_path = script_path.parent
print(f"\n7. Contenu du répertoire backend ({backend_path}):")
try:
    items = sorted(backend_path.iterdir())
    for item in items:
        item_type = "📁" if item.is_dir() else "📄"
        print(f"   {item_type} {item.name}")
except Exception as e:
    print(f"   Erreur: {e}")

print("\n" + "=" * 60)
print("FIN DU DIAGNOSTIC")
print("=" * 60)
