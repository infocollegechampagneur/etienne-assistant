#!/bin/bash
# Script de diagnostic pour Render

echo "=== Diagnostic Backend Etienne ==="
echo "Date: $(date)"
echo ""

echo "=== Variables d'environnement critiques ==="
echo "ADMIN_USERNAME: ${ADMIN_USERNAME:-NOT_SET}"
echo "ADMIN_PASSWORD_HASH présent: $([ -n "$ADMIN_PASSWORD_HASH" ] && echo "OUI" || echo "NON")"
echo "SMTP_HOST: ${SMTP_HOST:-NOT_SET}"
echo ""

echo "=== Test import security_advanced ==="
python3 -c "import security_advanced; print('✅ Import OK')" || echo "❌ Import FAILED"
echo ""

echo "=== Test import server ==="
python3 -c "import server; print('✅ Server import OK')" || echo "❌ Server import FAILED"
echo ""

echo "=== Dépendances clés installées ==="
pip list | grep -E "jose|passlib|reportlab|fastapi" || echo "Packages manquants!"
echo ""

echo "=== Fin diagnostic ==="
