# Supabase y CamS

El portal usa PostgreSQL con psycopg; no usa Supabase Auth ni expone credenciales de base en React.

Migración del portal: `migrations/202611020001_portal_schema.sql`. También se conserva la migración antigua de `crossing_events`, que pertenece al cliente anterior y no reemplaza las cuatro tablas del portal.

Configura `SUPABASE_DB_URL` en el `.env` privado de la raíz. Luego ejecuta:

```powershell
.\.venv-portal\Scripts\python.exe scripts/setup_database.py --require-supabase
```

El comando crea y verifica `cams_users`, `cams_sessions`, `cams_persons`, `cams_events`, con RLS y roles públicos sin acceso. Sin DSN válido devuelve error; no simula conexión a Supabase. Para comprobación sin DDL, agrega `--check-only`.

Consulta [SYSTEM.md](../SYSTEM.md) y la [documentación de conexión de Supabase](https://supabase.com/docs/guides/database/connecting-to-postgres). Los datos locales no se migran automáticamente al cambiar la conexión.
