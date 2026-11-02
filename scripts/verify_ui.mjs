// Run against the local stack. Uses a synthetic video, never the real webcam.
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';

const root = resolve(import.meta.dirname, '..');
const require = createRequire(resolve(root, '.runtime/browser/package.json'));
const { chromium } = require('playwright');
const artifacts = resolve(root, 'apps/web/artifacts');
await mkdir(artifacts, { recursive: true });
const browser = await chromium.launch({ args: [
  '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
  `--use-file-for-fake-video-capture=${resolve(root, '.runtime/camera-ana.y4m')}`,
] });
const context = await browser.newContext({ viewport: { width: 1440, height: 1040 }, permissions: ['camera'] });
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const email = `qa-${randomUUID()}@example.test`;
const password = randomUUID();
try {
  await page.goto('http://127.0.0.1:5173');
  await page.getByRole('heading', {name: /Tu espacio/}).waitFor();
  await page.screenshot({ path: resolve(artifacts, '01-inicio.png'), fullPage: true });
  await page.getByRole('button', {name: /^Iniciar sesión$/}).click();
  await page.screenshot({ path: resolve(artifacts, '02-iniciar-sesion.png'), fullPage: true });
  await page.getByRole('button', {name: 'Crea una cuenta'}).click();
  await page.screenshot({ path: resolve(artifacts, '03-crear-cuenta.png'), fullPage: true });
  await page.locator('input[name=name]').fill('Prueba de interfaz');
  await page.locator('input[name=email]').fill(email);
  await page.locator('input[name=password]').fill(password);
  await page.getByRole('button', {name: 'Crear mi cuenta'}).click();
  await page.getByRole('heading', {name: /^Registro de personas/}).waitFor();
  const token = await page.evaluate(() => localStorage.getItem('cams.token'));
  const me = await context.request.get('http://127.0.0.1:5173/api/portal/me', { headers: { Authorization: `Bearer ${token}` } });
  const account = await me.json();
  await writeFile(resolve(root, '.runtime/qa-account.json'), JSON.stringify({ id: account.id, email }));
  await page.getByRole('button', {name: 'Activar cámara'}).click();
  await page.getByRole('button', {name: 'Capturar fotografía'}).waitFor();
  await page.getByRole('button', {name: 'Capturar fotografía'}).click();
  await page.locator('input[name=full_name]').fill('Ana sintética');
  await page.locator('input[name=external_id]').fill('ANA-QA');
  await page.getByRole('button', {name: 'Registrar persona', exact: true}).click();
  await page.getByText('Ana sintética se registró correctamente.').first().waitFor({timeout: 30000});
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: resolve(artifacts, '04-registro.png'), fullPage: true });
  await page.getByRole('button', {name: 'Ampliar panel'}).click();
  await page.getByRole('button', {name: 'Reducir panel'}).waitFor();
  await page.getByRole('button', {name: 'Reducir panel'}).click();
  await page.getByRole('button', {name: 'Módulo 2 · Control de acceso'}).click();
  await page.getByRole('button', {name: 'Activar cámara'}).click();
  await page.getByRole('button', {name: 'Iniciar reconocimiento'}).click();
  await page.getByText('¡Fuiste reconocido! Ana sintética.').waitFor({timeout: 60000});
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: resolve(artifacts, '05-control-acceso.png'), fullPage: true });
  await page.getByRole('button', {name: 'Contraer barra lateral'}).click();
  await page.getByRole('button', {name: 'Expandir barra lateral'}).waitFor();
  await page.getByRole('button', {name: 'Cerrar sesión'}).click();
  await page.getByRole('heading', {name: 'Qué bueno verte de nuevo.'}).waitFor();
  await page.locator('input[name=email]').fill(email);
  await page.locator('input[name=password]').fill(password);
  await page.getByRole('button', {name: /^Iniciar sesión$/}).click();
  await page.getByRole('heading', {name: /^Registro de personas/}).waitFor();
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({ path: resolve(artifacts, '06-movil.png'), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  if (overflow) throw new Error('La vista móvil tiene desbordamiento horizontal');
  if (errors.length) throw new Error(errors.join('\n'));
  console.log('BROWSER_OK: inicio, login, cuenta, cámara sintética, registro, reconocimiento real, paneles, logout y móvil');
} finally {
  await browser.close();
}
