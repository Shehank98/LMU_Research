async function request(url, options) {
  const res = await fetch(url, options)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)
  return data
}

export const getModelStatus = () => request('/api/models/status')
export const getResults = () => request('/api/results')

export function analyseImage(file) {
  const body = new FormData()
  body.append('file', file)
  return request('/api/predict', { method: 'POST', body })
}
