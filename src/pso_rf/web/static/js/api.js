// Backend API client. Every result shown in the UI comes from these endpoints (saved files or a live run).

async function request(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
  return body;
}

let metaPromise = null;
export const api = {
  meta: () => (metaPromise ??= request("/api/meta")),
  results: (exp) => request(`/api/results${exp ? `?exp=${encodeURIComponent(exp)}` : ""}`),
  replay: (exp, dataset, fold) =>
    request(`/api/replay?${new URLSearchParams({ ...(exp ? { exp } : {}), dataset, fold })}`),
  landscape: (dataset) => request(`/api/landscape?dataset=${dataset}`),
  isolation: () => request("/api/proof/isolation", { method: "POST" }),
  startLive: (params) =>
    request("/api/live", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) }),
};

/**
 * Subscribe to a live job's events. Reconnects from the last received index, so no event is lost or duplicated.
 * Returns a function that stops listening.
 */
export function streamLive(id, onEvent, onEnd, onGone) {
  let received = 0, source = null, stopped = false;
  const connect = () => {
    source = new EventSource(`/api/live/${id}/events?from=${received}`);
    source.onmessage = (m) => { received += 1; onEvent(JSON.parse(m.data)); };
    source.addEventListener("end", () => { source.close(); if (!stopped) onEnd && onEnd(); });
    source.addEventListener("gone", () => { stopped = true; source.close(); onGone && onGone(); });
    source.onerror = () => { source.close(); if (!stopped) setTimeout(connect, 800); };
  };
  connect();
  return () => { stopped = true; source && source.close(); };
}
