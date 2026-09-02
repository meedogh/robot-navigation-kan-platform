import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function getJSON<T>(path: string): Promise<T> {
  const { data } = await axios.get<T>(`${API_BASE_URL}${path}`);
  return data;
}
