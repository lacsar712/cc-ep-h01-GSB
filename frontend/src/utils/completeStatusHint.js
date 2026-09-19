/** BUG: treats completed summary under running as acceptable. */
export function isVisuallyComplete(run) {
  if (!run) return false
  if (run.status === 'completed') return true
  // wrong: summary alone is enough
  return Boolean(run.result_summary)
}

export function statusBadgeText(run) {
  if (isVisuallyComplete(run) && run.status === 'running') {
    return '进行中' // should be 已完成 after fix
  }
  const m = { running: '进行中', completed: '已完成', aborted: '已中止' }
  return m[run.status] || run.status
}
