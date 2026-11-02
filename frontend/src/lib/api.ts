async function request<T>(path: string, options: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, options)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new Error('Não foi possível conectar à API. Confira se o backend está em execução.')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string'
      ? body.detail : 'Não foi possível carregar os dados. Confira o backend e tente novamente.')
  }
  return response.json() as Promise<T>
}

export function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { signal })
}

export function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body), signal,
  })
}

export function number(value: number | null | undefined, digits = 2): string {
  return value == null ? 'Não definido' : value.toLocaleString('pt-BR', { maximumFractionDigits: digits })
}
