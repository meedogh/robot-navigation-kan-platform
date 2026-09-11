import axios from "axios";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function getJSON<T>(path: string): Promise<T> {
  const { data } = await axios.get<T>(`${API_BASE_URL}${path}`);
  return data;
}

export async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const { data } = await axios.post<T>(`${API_BASE_URL}${path}`, body ?? {});
  return data;
}

export async function deleteJSON<T = void>(path: string): Promise<T> {
  const { data } = await axios.delete<T>(`${API_BASE_URL}${path}`);
  return data;
}

// Download a (possibly binary) file from the backend, surfacing API error
// details (e.g. "ONNX export failed: no checkpoint") to the caller.
export async function downloadFile(
  path: string,
  fallbackFilename: string
): Promise<{ ok: boolean; filename?: string; error?: string }> {
  try {
    const response = await axios.get(`${API_BASE_URL}${path}`, {
      responseType: "blob",
    });

    // Prefer the server-provided filename from Content-Disposition.
    const disposition: string = response.headers["content-disposition"] ?? "";
    const match = disposition.match(/filename="?([^";]+)"?/);
    const filename = match?.[1] ?? fallbackFilename;

    const url = window.URL.createObjectURL(response.data);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.URL.revokeObjectURL(url);

    return { ok: true, filename };
  } catch (e: any) {
    let error = e.message;
    if (e?.response?.data) {
      try {
        const text = await e.response.data.text();
        const parsed = JSON.parse(text);
        error = parsed.detail ?? text;
      } catch {
        // keep the axios message
      }
    }
    return { ok: false, error };
  }
}
