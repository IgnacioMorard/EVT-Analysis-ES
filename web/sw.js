/*
 * Service Worker — cache de archivos locales del proyecto.
 *
 * QUE CACHEA:
 *   - index.html, evt_core.py, worker.js, sw.js (archivos locales)
 *
 * QUE NO CACHEA:
 *   - Pyodide y sus paquetes (~60 MB desde cdn.jsdelivr.net).
 *     Son respuestas cross-origin (response.type === "cors"),
 *     y este SW solo cachea respuestas "basic" (mismo origen).
 *     El browser mantiene su propia cache HTTP para el CDN,
 *     pero no garantiza persistencia offline.
 *
 *   - Google Fonts (fonts.googleapis.com / fonts.gstatic.com).
 *
 * CONSECUENCIA:
 *   La app NO funciona 100% offline. Necesita conexion para
 *   descargar Pyodide en cada sesion nueva (o depende de la
 *   cache HTTP del browser, que puede purgarse).
 */

const CACHE_NAME = "evt-v2";
const PRECACHE = [
  "./",
  "./index.html",
  "./evt_core.py",
  "./worker.js",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  e.respondWith(
    caches.match(e.request).then((cached) => {
      if (cached) return cached;
      return fetch(e.request).then((response) => {
        // Solo cachear respuestas del mismo origen (no CDN cross-origin)
        if (response && response.status === 200 && response.type === "basic") {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(e.request, clone));
        }
        return response;
      }).catch(() => cached);
    })
  );
});
