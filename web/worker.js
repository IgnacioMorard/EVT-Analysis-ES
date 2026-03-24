/*
 * Web Worker: runs Pyodide + evt_core.py off the main thread.
 *
 * Messages IN  (from main):
 *   { type: "init" }
 *   { type: "run",  csvText: string, config: object }
 *
 * Messages OUT (to main):
 *   { type: "init-progress", msg: string }
 *   { type: "ready" }
 *   { type: "result", data: object }
 *   { type: "error",  message: string }
 */

let pyodide = null;
let evtCode = null;

async function initPyodide() {
  postMessage({ type: "init-progress", msg: "Descargando Pyodide..." });
  importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js");

  postMessage({ type: "init-progress", msg: "Inicializando Python..." });
  pyodide = await loadPyodide();

  postMessage({ type: "init-progress", msg: "Instalando numpy, pandas, scipy, matplotlib..." });
  await pyodide.loadPackage(["numpy", "pandas", "scipy", "matplotlib"]);

  postMessage({ type: "init-progress", msg: "Cargando modulo EVT..." });
  const resp = await fetch("./evt_core.py");
  evtCode = await resp.text();
  await pyodide.runPythonAsync(evtCode);

  postMessage({ type: "ready" });
}

async function runAnalysis(csvText, config) {
  try {
    // Put CSV text and config into Python namespace
    pyodide.globals.set("_csv_text", csvText);
    pyodide.globals.set("_config_js", pyodide.toPy(config));

    const resultProxy = await pyodide.runPythonAsync(`
import json, math

_cfg = dict(_config_js) if _config_js else {}
_result = run_analysis(_csv_text, _cfg)

# Convert to JSON-safe dict
def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    return obj

json.dumps(_to_json_safe(_result))
`);

    const jsonStr = resultProxy;
    const data = JSON.parse(jsonStr);
    postMessage({ type: "result", data });
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  }
}

onmessage = async (e) => {
  const { type } = e.data;
  if (type === "init") {
    try {
      await initPyodide();
    } catch (err) {
      postMessage({ type: "error", message: "Error inicializando Pyodide: " + String(err) });
    }
  } else if (type === "run") {
    await runAnalysis(e.data.csvText, e.data.config);
  }
};
