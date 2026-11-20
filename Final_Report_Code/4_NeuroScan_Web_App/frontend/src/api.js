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

export const getPatients = () => request('/api/patients')
export const getPatient = (id) => request(`/api/patients/${id}`)

export const createPatient = (patient) =>
  request('/api/patients', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patient),
  })

export function uploadPatientScan(id, file, study, notes) {
  const body = new FormData()
  body.append('file', file)
  body.append('study', study || '')
  body.append('notes', notes || '')
  return request(`/api/patients/${id}/scans`, { method: 'POST', body })
}

export const reviewScan = (id, scanId, review) =>
  request(`/api/patients/${id}/scans/${scanId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(review),
  })
