import { useState, useEffect, useCallback } from 'react'
import { portkeyLogsApi, PortkeyLog } from '../api/client'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'

// Format date for display
function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return '-'
  try {
    return new Date(dateStr).toLocaleString()
  } catch {
    return dateStr
  }
}

// Format relative time
function formatRelativeTime(dateStr: string | undefined): string {
  if (!dateStr) return 'Never'
  try {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    const diffDays = Math.floor(diffHours / 24)
    return `${diffDays}d ago`
  } catch {
    return dateStr
  }
}

// Format cost
function formatCost(cost: number | undefined): string {
  if (cost === undefined || cost === null) return '-'
  return `$${cost.toFixed(6)}`
}

// Format latency
function formatLatency(ms: number | undefined): string {
  if (ms === undefined || ms === null) return '-'
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

// Status badge variant
function getStatusVariant(isSuccess: boolean | undefined, status: string | undefined): 'success' | 'error' | 'warning' | 'info' {
  if (isSuccess === true) return 'success'
  if (isSuccess === false) return 'error'
  if (status?.toLowerCase().includes('error')) return 'error'
  if (status?.toLowerCase().includes('success')) return 'success'
  return 'info'
}

// Log detail modal
function LogDetailModal({ 
  log, 
  onClose 
}: { 
  log: PortkeyLog | null
  onClose: () => void 
}) {
  if (!log) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        <div className="p-4 border-b border-slate-700 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-white">Log Details</h2>
          <button 
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="p-4 overflow-y-auto flex-1 space-y-4">
          {/* Basic Info */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">ID</p>
              <p className="text-slate-200 font-mono text-sm truncate">{log.id}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Model</p>
              <p className="text-slate-200">{log.ai_model || '-'}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Provider</p>
              <p className="text-slate-200">{log.ai_org || log.ai_provider || '-'}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Created</p>
              <p className="text-slate-200">{formatDate(log.time_of_generation || log.created_at)}</p>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 bg-slate-800/50 rounded-lg p-4">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Cost</p>
              <p className="text-emerald-400 font-semibold">{formatCost(log.cost)}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Latency</p>
              <p className="text-blue-400 font-semibold">{formatLatency(log.response_time)}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Input Tokens</p>
              <p className="text-slate-200">{(log.req_units || log.prompt_tokens)?.toLocaleString() || '-'}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Output Tokens</p>
              <p className="text-slate-200">{(log.res_units || log.completion_tokens)?.toLocaleString() || '-'}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Total Tokens</p>
              <p className="text-slate-200">{(log.total_units || log.total_tokens)?.toLocaleString() || '-'}</p>
            </div>
          </div>

          {/* Request */}
          {log.request && (
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-2">Request</h3>
              <pre className="bg-slate-950 p-4 rounded-lg overflow-x-auto text-xs text-slate-300 font-mono">
                {JSON.stringify(log.request, null, 2)}
              </pre>
            </div>
          )}

          {/* Response */}
          {log.response && (
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-2">Response</h3>
              <pre className="bg-slate-950 p-4 rounded-lg overflow-x-auto text-xs text-slate-300 font-mono">
                {JSON.stringify(log.response, null, 2)}
              </pre>
            </div>
          )}

          {/* Metadata */}
          {log.metadata && Object.keys(log.metadata).length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-2">Metadata</h3>
              <pre className="bg-slate-950 p-4 rounded-lg overflow-x-auto text-xs text-slate-300 font-mono">
                {JSON.stringify(log.metadata, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function PortkeyLogs() {
  const [logs, setLogs] = useState<PortkeyLog[]>([])
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hours, setHours] = useState(168) // 7 days for refresh
  const [limit, setLimit] = useState(100)
  const [selectedLog, setSelectedLog] = useState<PortkeyLog | null>(null)
  const [workspaceId, setWorkspaceId] = useState('2d469afe-6e46-4929-ab71-21de003b711d')
  const [fromCache, setFromCache] = useState(false)
  const [lastSynced, setLastSynced] = useState<string | undefined>()

  // Load logs from cache (database)
  const loadFromCache = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await portkeyLogsApi.getLogs({
        workspace_id: workspaceId || undefined,
        limit,
        refresh: false, // Load from cache only
      })
      setLogs(response.logs)
      setFromCache(response.from_cache)
      setLastSynced(response.last_synced)
    } catch (err) {
      console.error('Failed to load logs:', err)
      setError(err instanceof Error ? err.message : 'Failed to load logs from cache')
    } finally {
      setLoading(false)
    }
  }, [limit, workspaceId])

  // Refresh logs from Portkey API
  const refreshFromPortkey = useCallback(async () => {
    setRefreshing(true)
    setError(null)
    
    try {
      const response = await portkeyLogsApi.getLogs({
        workspace_id: workspaceId || undefined,
        hours,
        limit,
        refresh: true, // Fetch new logs from Portkey
      })
      setLogs(response.logs)
      setFromCache(response.from_cache)
      setLastSynced(response.last_synced)
    } catch (err) {
      console.error('Failed to refresh logs:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch logs from Portkey')
    } finally {
      setRefreshing(false)
    }
  }, [hours, limit, workspaceId])

  // Auto-load from cache on mount
  useEffect(() => {
    loadFromCache()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Calculate stats
  const stats = {
    total: logs.length,
    success: logs.filter(l => l.is_success === true).length,
    failed: logs.filter(l => l.is_success === false).length,
    totalCost: logs.reduce((sum, l) => sum + (l.cost || 0), 0),
    avgLatency: logs.length > 0 
      ? logs.reduce((sum, l) => sum + (l.response_time || 0), 0) / logs.length 
      : 0,
    totalTokens: logs.reduce((sum, l) => sum + (l.total_units || l.total_tokens || 0), 0),
  }

  // Get unique models
  const models = [...new Set(logs.map(l => l.ai_model).filter(Boolean))]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">
            Portkey Logs
          </h1>
          <p className="text-slate-400 mt-1">
            View and analyze logs from your Portkey workspace
          </p>
        </div>
        
        {/* Cache Status */}
        <div className="text-right">
          <div className="flex items-center gap-2 justify-end">
            {fromCache ? (
              <Badge variant="info">
                <span className="flex items-center gap-1">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M3 12v3c0 1.657 3.134 3 7 3s7-1.343 7-3v-3c0 1.657-3.134 3-7 3s-7-1.343-7-3z"/>
                    <path d="M3 7v3c0 1.657 3.134 3 7 3s7-1.343 7-3V7c0 1.657-3.134 3-7 3S3 8.657 3 7z"/>
                    <path d="M17 5c0 1.657-3.134 3-7 3S3 6.657 3 5s3.134-3 7-3 7 1.343 7 3z"/>
                  </svg>
                  From Cache
                </span>
              </Badge>
            ) : (
              <Badge variant="success">
                <span className="flex items-center gap-1">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd"/>
                  </svg>
                  Just Synced
                </span>
              </Badge>
            )}
          </div>
          {lastSynced && (
            <p className="text-xs text-slate-500 mt-1">
              Last synced: {formatRelativeTime(lastSynced)}
            </p>
          )}
        </div>
      </div>

      {/* Controls */}
      <Card>
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-1">
              Workspace ID
            </label>
            <input
              type="text"
              value={workspaceId}
              onChange={(e) => setWorkspaceId(e.target.value)}
              className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-slate-200 w-80 focus:outline-none focus:ring-2 focus:ring-violet-500"
              placeholder="Enter workspace ID"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-1">
              Refresh Time Range
            </label>
            <select
              value={hours}
              onChange={(e) => setHours(Number(e.target.value))}
              className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              <option value={1}>Last 1 hour</option>
              <option value={6}>Last 6 hours</option>
              <option value={12}>Last 12 hours</option>
              <option value={24}>Last 24 hours</option>
              <option value={48}>Last 48 hours</option>
              <option value={72}>Last 3 days</option>
              <option value={168}>Last 7 days</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-400 mb-1">
              Max Logs
            </label>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>500</option>
              <option value={1000}>1000</option>
            </select>
          </div>

          <div className="flex gap-2">
            <Button 
              onClick={loadFromCache} 
              disabled={loading || refreshing}
              variant="secondary"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Loading...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                  Load from Cache
                </>
              )}
            </Button>

            <Button 
              onClick={refreshFromPortkey} 
              disabled={loading || refreshing}
              variant="primary"
            >
              {refreshing ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Syncing...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Refresh from Portkey
                </>
              )}
            </Button>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            {error}
          </div>
        )}
      </Card>

      {/* Stats */}
      {logs.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <Card className="text-center">
            <p className="text-2xl font-bold text-white">{stats.total}</p>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Total Logs</p>
          </Card>
          <Card className="text-center">
            <p className="text-2xl font-bold text-emerald-400">{stats.success}</p>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Success</p>
          </Card>
          <Card className="text-center">
            <p className="text-2xl font-bold text-red-400">{stats.failed}</p>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Failed</p>
          </Card>
          <Card className="text-center">
            <p className="text-2xl font-bold text-amber-400">{formatCost(stats.totalCost)}</p>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Total Cost</p>
          </Card>
          <Card className="text-center">
            <p className="text-2xl font-bold text-blue-400">{formatLatency(stats.avgLatency)}</p>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Avg Latency</p>
          </Card>
          <Card className="text-center">
            <p className="text-2xl font-bold text-violet-400">{stats.totalTokens.toLocaleString()}</p>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Total Tokens</p>
          </Card>
        </div>
      )}

      {/* Models used */}
      {models.length > 0 && (
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-2">Models Used</h3>
          <div className="flex flex-wrap gap-2">
            {models.map(model => (
              <Badge key={model} variant="info">{model}</Badge>
            ))}
          </div>
        </Card>
      )}

      {/* Logs Table */}
      {logs.length > 0 && (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Provider
                  </th>
                  <th className="text-left py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="text-right py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Tokens
                  </th>
                  <th className="text-right py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Cost
                  </th>
                  <th className="text-right py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Latency
                  </th>
                  <th className="text-center py-3 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {logs.map((log) => (
                  <tr 
                    key={log.id} 
                    className="hover:bg-slate-800/50 transition-colors cursor-pointer"
                    onClick={() => setSelectedLog(log)}
                  >
                    <td className="py-3 px-4 text-sm text-slate-300">
                      {formatDate(log.time_of_generation || log.created_at)}
                    </td>
                    <td className="py-3 px-4 text-sm text-slate-200 font-medium">
                      {log.ai_model || '-'}
                    </td>
                    <td className="py-3 px-4 text-sm text-slate-400">
                      {log.ai_org || log.ai_provider || '-'}
                    </td>
                    <td className="py-3 px-4">
                      <Badge variant={getStatusVariant(log.is_success, log.status)}>
                        {log.is_success ? 'Success' : log.is_success === false ? 'Failed' : log.status || 'Unknown'}
                      </Badge>
                    </td>
                    <td className="py-3 px-4 text-sm text-slate-300 text-right font-mono">
                      {(log.total_units || log.total_tokens)?.toLocaleString() || '-'}
                    </td>
                    <td className="py-3 px-4 text-sm text-emerald-400 text-right font-mono">
                      {formatCost(log.cost)}
                    </td>
                    <td className="py-3 px-4 text-sm text-blue-400 text-right font-mono">
                      {formatLatency(log.response_time)}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setSelectedLog(log)
                        }}
                        className="text-slate-400 hover:text-violet-400 transition-colors"
                      >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Empty state */}
      {!loading && !refreshing && logs.length === 0 && (
        <Card className="text-center py-12">
          <div className="text-slate-500 mb-4">
            <svg className="w-16 h-16 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-slate-300 mb-2">
            No logs in cache
          </h3>
          <p className="text-slate-500 mb-4">
            Click "Refresh from Portkey" to fetch and store logs from your workspace
          </p>
          <Button onClick={refreshFromPortkey} variant="primary">
            Refresh from Portkey
          </Button>
        </Card>
      )}

      {/* Log Detail Modal */}
      <LogDetailModal 
        log={selectedLog} 
        onClose={() => setSelectedLog(null)} 
      />
    </div>
  )
}
