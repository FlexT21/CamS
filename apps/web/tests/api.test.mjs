import { test } from 'node:test';
import assert from 'node:assert/strict';
import { request, ApiError, asList, recognitionDelay, monitorSessionId } from '../src/api.js';

test('restarting recognition reuses the monitor session in the same tab', () => {
  const previous = globalThis.sessionStorage;
  const values = new Map();
  globalThis.sessionStorage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  try {
    const first = monitorSessionId();
    assert.match(first, /^[A-Za-z0-9_-]{1,64}$/);
    assert.equal(monitorSessionId(), first);
  } finally {
    if (previous === undefined) delete globalThis.sessionStorage;
    else globalThis.sessionStorage = previous;
  }
});

test('portal requests use the exact prefix, bearer header, JSON body and abort signal', async () => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };
  try {
    const controller = new AbortController();
    const body = { name: 'Ana López', external_id: 'A-101', image: 'data:image/jpeg;base64,abc' };
    await request('/persons', { token: 'test-token', method: 'POST', body, signal: controller.signal });
    assert.equal(calls[0].url, '/api/portal/persons');
    assert.equal(calls[0].options.headers.Authorization, 'Bearer test-token');
    assert.equal(calls[0].options.headers['Content-Type'], 'application/json');
    assert.deepEqual(JSON.parse(calls[0].options.body), body);
    assert.equal(calls[0].options.signal, controller.signal);
    await request('/auth/login', { method: 'POST', body: { email: 'ana@example.com', password: 'secret' } });
    assert.equal(calls[1].url, '/api/portal/auth/login');
    assert.equal(calls[1].options.headers.Authorization, undefined);
    await request('/monitor/frame', { token: 'test-token', method: 'POST', body: { image: body.image, session_id: 'session-1' } });
    assert.deepEqual(JSON.parse(calls[2].options.body), { image: body.image, session_id: 'session-1' });
  } finally { globalThis.fetch = original; }
});

test('server errors retain their status and actual validation messages', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: [{ msg: 'El identificador ya existe' }] }), { status: 422 });
    await assert.rejects(request('/persons'), error => error instanceof ApiError && error.status === 422 && error.message === 'El identificador ya existe');
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Sesión expirada' }), { status: 401 });
    await assert.rejects(request('/me'), error => error.status === 401 && error.message === 'Sesión expirada');
    globalThis.fetch = async () => new Response('<html>Bad gateway</html>', { status: 502 });
    await assert.rejects(request('/status'), error => error.status === 502 && !error.message.includes('<html>'));
  } finally { globalThis.fetch = original; }
});

test('invalid success responses and network failures cannot become success data', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response('not json', { status: 200 });
    await assert.rejects(request('/persons'), /JSON válida/);
    globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
    await assert.rejects(request('/stats'), TypeError);
  } finally { globalThis.fetch = original; }
});

test('list decoding rejects malformed data instead of showing a fake empty list', () => {
  const persons = [{ name: 'Ana' }];
  assert.deepEqual(asList(persons, 'persons'), persons);
  assert.deepEqual(asList({ persons }, 'persons'), persons);
  assert.throws(() => asList({ success: true }, 'persons'), /formato esperado/);
});

test('recognition waits 300ms after completion and backs off unavailable services', () => {
  assert.equal(recognitionDelay(), 300);
  assert.equal(recognitionDelay(1), 1000);
  assert.equal(recognitionDelay(2), 2000);
  assert.equal(recognitionDelay(3), 4000);
  assert.equal(recognitionDelay(10), 15000);
});
