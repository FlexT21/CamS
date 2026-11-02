"""Remove only the local throwaway account identified by browser QA."""
import json
from pathlib import Path
import sqlite3

root = Path(__file__).resolve().parents[1]
marker = root / '.runtime/qa-account.json'
if marker.exists():
    account = json.loads(marker.read_text(encoding='utf-8'))
    if not (account['email'].startswith('qa-') and account['email'].endswith('@example.test')):
        raise SystemExit('No es una cuenta temporal de QA')
    connection = sqlite3.connect(root / '.runtime/portal.sqlite3')
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('DELETE FROM cams_users WHERE id=? AND email=?', (account['id'], account['email']))
    connection.commit()
    connection.close()
    marker.unlink()
    print('Cuenta temporal de prueba eliminada; las cuentas de usuario se conservan')
