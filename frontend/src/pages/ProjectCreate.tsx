import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { ArrowLeft, Sparkles, Database, RefreshCw, Check, Target } from 'lucide-react'
import Card from '../components/Card'
import Button from '../components/Button'
import Badge from '../components/Badge'
import { projectsApi, portkeyLogsApi, PortkeyLog } from '../api/client'

// LiveBench capability dimension info
const LIVEBENCH_DIMENSIONS = [
  { key: 'reasoning', label: 'Reasoning', description: 'Logical reasoning and problem-solving ability' },
  { key: 'coding', label: 'Coding', description: 'Code generation, review, and debugging' },
  { key: 'agentic_coding', label: 'Agentic Coding', description: 'Autonomous coding with tool use' },
  { key: 'mathematics', label: 'Mathematics', description: 'Mathematical problem-solving' },
  { key: 'data_analysis', label: 'Data Analysis', description: 'Data interpretation and analysis' },
  { key: 'language', label: 'Language', description: 'Language understanding and generation' },
  { key: 'instruction_following', label: 'Instruction Following', description: 'Following complex instructions accurately' },
] as const

type CapabilityExpectations = {
  [K in typeof LIVEBENCH_DIMENSIONS[number]['key']]?: number | null
}

// Format date for display
function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return '-'
  try {
    return new Date(dateStr).toLocaleString()
  } catch {
    return dateStr
  }
}

// Format cost
function formatCost(cost: number | undefined): string {
  if (cost === undefined || cost === null) return '-'
  return `$${cost.toFixed(4)}`
}

export default function ProjectCreate() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    name: '',
    agent_purpose: '',
    description: '',
    current_model: '',
    current_provider: '',
    selected_log_ids: [] as string[],
  })
  
  // Capability expectations state (LiveBench minimum scores)
  const [capabilityExpectations, setCapabilityExpectations] = useState<CapabilityExpectations>({})
  const [showCapabilities, setShowCapabilities] = useState(false)

  // Update a capability expectation
  const updateCapability = (key: string, value: number | null) => {
    setCapabilityExpectations(prev => ({
      ...prev,
      [key]: value
    }))
  }

  // Check if any capability is set
  const hasCapabilities = Object.values(capabilityExpectations).some(v => v !== null && v !== undefined)

  // Logs state
  const [logs, setLogs] = useState<PortkeyLog[]>([])
  const [logsLoading, setLogsLoading] = useState(false)
  const [logsError, setLogsError] = useState<string | null>(null)
  const [logsFetched, setLogsFetched] = useState(false)
  
  // User filter state
  const [users, setUsers] = useState<string[]>([])
  const [selectedUser, setSelectedUser] = useState<string>('')
  const [usersLoading, setUsersLoading] = useState(false)

  // Load available users for filter
  const loadUsers = useCallback(async () => {
    setUsersLoading(true)
    try {
      const userList = await portkeyLogsApi.getUsers()
      setUsers(userList)
    } catch (err) {
      console.error('Failed to load users:', err)
    } finally {
      setUsersLoading(false)
    }
  }, [])

  // Fetch logs from Portkey
  const fetchLogs = useCallback(async () => {
    setLogsLoading(true)
    setLogsError(null)
    
    try {
      const response = await portkeyLogsApi.getLogs({
        hours: 168, // Last 7 days
        limit: 500,
        user_filter: selectedUser || undefined,
      })
      setLogs(response.logs)
      setLogsFetched(true)
    } catch (err) {
      console.error('Failed to fetch logs:', err)
      setLogsError(err instanceof Error ? err.message : 'Failed to fetch logs')
    } finally {
      setLogsLoading(false)
    }
  }, [selectedUser])

  // Auto-fetch users and logs on mount
  useEffect(() => {
    loadUsers()
    fetchLogs()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Refetch logs when user filter changes
  useEffect(() => {
    if (logsFetched) {
      fetchLogs()
    }
  }, [selectedUser]) // eslint-disable-line react-hooks/exhaustive-deps

  // Toggle log selection
  const toggleLogSelection = (logId: string) => {
    setFormData(prev => ({
      ...prev,
      selected_log_ids: prev.selected_log_ids.includes(logId)
        ? prev.selected_log_ids.filter(id => id !== logId)
        : [...prev.selected_log_ids, logId]
    }))
  }

  // Select all logs
  const selectAllLogs = () => {
    setFormData(prev => ({
      ...prev,
      selected_log_ids: logs.map(log => log.id)
    }))
  }

  // Deselect all logs
  const deselectAllLogs = () => {
    setFormData(prev => ({
      ...prev,
      selected_log_ids: []
    }))
  }

  const createMutation = useMutation({
    mutationFn: (data: { formData: typeof formData, capabilities: CapabilityExpectations }) => {
      // Build capability expectations object, filtering out null/undefined values
      const capabilityData = Object.entries(data.capabilities).reduce((acc, [key, value]) => {
        if (value !== null && value !== undefined) {
          acc[key] = value
        }
        return acc
      }, {} as Record<string, number>)
      
      return projectsApi.create({
        ...data.formData,
        selected_log_ids: data.formData.selected_log_ids.length > 0 ? data.formData.selected_log_ids : undefined,
        capability_expectations: Object.keys(capabilityData).length > 0 ? capabilityData : undefined,
      })
    },
    onSuccess: (project) => {
      navigate(`/projects/${project.id}`)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createMutation.mutate({ formData, capabilities: capabilityExpectations })
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <button 
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 text-void-400 hover:text-void-200 mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <h1 className="text-2xl font-display font-bold text-void-50">Create New Project</h1>
        <p className="text-void-400 mt-1">Set up a new AI agent for model optimization</p>
      </div>

      {/* Form */}
      <Card>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Info */}
          <div>
            <h3 className="font-semibold text-void-100 mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-accent-500" />
              Basic Information
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Project Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., Customer Support Bot"
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Agent Purpose *
                </label>
                <textarea
                  value={formData.agent_purpose}
                  onChange={(e) => setFormData({ ...formData, agent_purpose: e.target.value })}
                  placeholder="Describe what this AI agent does and its primary use cases..."
                  rows={3}
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500 resize-none"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Additional notes about this project..."
                  rows={2}
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500 resize-none"
                />
              </div>
            </div>
          </div>

          {/* Log Selection */}
          <div>
            <h3 className="font-semibold text-void-100 mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-emerald-500" />
              Associate Portkey Logs
            </h3>
            <p className="text-sm text-void-400 mb-4">
              Select logs from your Portkey workspace to import and associate with this project. 
              Selected logs will be stored locally for analysis by the analytics engine.
            </p>
            
            {/* Logs Controls */}
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={fetchLogs}
                disabled={logsLoading}
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${logsLoading ? 'animate-spin' : ''}`} />
                {logsLoading ? 'Loading...' : 'Refresh Logs'}
              </Button>
              
              {/* User Filter */}
              <div className="flex items-center gap-2">
                <label className="text-sm text-void-400">Filter by User:</label>
                <select
                  value={selectedUser}
                  onChange={(e) => setSelectedUser(e.target.value)}
                  className="bg-void-800 border border-void-700 rounded-lg px-3 py-1.5 text-sm text-void-200 focus:outline-none focus:border-accent-500 min-w-[140px]"
                  disabled={usersLoading}
                >
                  <option value="">All Users</option>
                  {users.map(user => (
                    <option key={user} value={user}>{user}</option>
                  ))}
                </select>
              </div>
              
              {logs.length > 0 && (
                <>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={selectAllLogs}
                  >
                    Select All
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={deselectAllLogs}
                  >
                    Deselect All
                  </Button>
                  <span className="text-sm text-void-400 ml-auto">
                    {formData.selected_log_ids.length} of {logs.length} selected
                  </span>
                </>
              )}
            </div>

            {/* Error State */}
            {logsError && (
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm mb-4">
                {logsError}
              </div>
            )}

            {/* Logs List */}
            {logsFetched && logs.length === 0 && !logsLoading && (
              <div className="p-6 bg-void-800/50 rounded-lg text-center">
                <p className="text-void-400">No logs found in your Portkey workspace.</p>
                <p className="text-void-500 text-sm mt-1">
                  Make some API calls through Portkey first, then refresh.
                </p>
              </div>
            )}

            {logs.length > 0 && (
              <div className="border border-void-700 rounded-lg overflow-hidden max-h-80 overflow-y-auto">
                <table className="w-full">
                  <thead className="bg-void-800 sticky top-0">
                    <tr>
                      <th className="w-10 px-3 py-2"></th>
                      <th className="text-left px-3 py-2 text-xs font-semibold text-void-400 uppercase">Time</th>
                      <th className="text-left px-3 py-2 text-xs font-semibold text-void-400 uppercase">Model</th>
                      <th className="text-left px-3 py-2 text-xs font-semibold text-void-400 uppercase">Provider</th>
                      <th className="text-right px-3 py-2 text-xs font-semibold text-void-400 uppercase">Cost</th>
                      <th className="text-center px-3 py-2 text-xs font-semibold text-void-400 uppercase">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-void-800">
                    {logs.map((log) => {
                      const isSelected = formData.selected_log_ids.includes(log.id)
                      return (
                        <tr 
                          key={log.id}
                          onClick={() => toggleLogSelection(log.id)}
                          className={`cursor-pointer transition-colors ${
                            isSelected 
                              ? 'bg-accent-500/10 hover:bg-accent-500/20' 
                              : 'hover:bg-void-800/50'
                          }`}
                        >
                          <td className="px-3 py-2">
                            <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${
                              isSelected 
                                ? 'bg-accent-500 border-accent-500' 
                                : 'border-void-600 bg-void-800'
                            }`}>
                              {isSelected && <Check className="w-3 h-3 text-white" />}
                            </div>
                          </td>
                          <td className="px-3 py-2 text-sm text-void-300">
                            {formatDate(log.time_of_generation || log.created_at)}
                          </td>
                          <td className="px-3 py-2 text-sm text-void-200 font-medium">
                            {log.ai_model || '-'}
                          </td>
                          <td className="px-3 py-2 text-sm text-void-400">
                            {log.ai_org || log.ai_provider || '-'}
                          </td>
                          <td className="px-3 py-2 text-sm text-emerald-400 text-right font-mono">
                            {formatCost(log.cost)}
                          </td>
                          <td className="px-3 py-2 text-center">
                            <Badge variant={log.is_success ? 'success' : log.is_success === false ? 'error' : 'info'}>
                              {log.is_success ? 'OK' : log.is_success === false ? 'Fail' : '?'}
                            </Badge>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Current Model */}
          <div>
            <h3 className="font-semibold text-void-100 mb-4">Current Model Configuration</h3>
            <p className="text-sm text-void-400 mb-4">
              Optionally specify the model your agent currently uses.
            </p>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Provider
                </label>
                <input
                  type="text"
                  value={formData.current_provider}
                  onChange={(e) => setFormData({ ...formData, current_provider: e.target.value })}
                  placeholder="e.g., openai"
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-void-300 mb-2">
                  Model
                </label>
                <input
                  type="text"
                  value={formData.current_model}
                  onChange={(e) => setFormData({ ...formData, current_model: e.target.value })}
                  placeholder="e.g., gpt-4o"
                  className="w-full px-4 py-2 bg-void-800 border border-void-700 rounded-lg text-void-100 placeholder-void-500 focus:outline-none focus:border-accent-500"
                />
              </div>
            </div>
          </div>

          {/* Capability Expectations (LiveBench) */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-semibold text-void-100 flex items-center gap-2">
                  <Target className="w-5 h-5 text-purple-500" />
                  Capability Expectations
                </h3>
                <p className="text-sm text-void-400 mt-1">
                  Set minimum benchmark scores based on LiveBench. Models not meeting these will be filtered out.
                </p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setShowCapabilities(!showCapabilities)}
              >
                {showCapabilities ? 'Hide' : 'Configure'}
              </Button>
            </div>
            
            {showCapabilities && (
              <div className="space-y-4 p-4 bg-void-800/50 rounded-lg border border-void-700">
                <p className="text-xs text-void-500 mb-4">
                  Drag sliders to set minimum scores (0-100). Leave at 0 to skip that dimension.
                  Based on <a href="https://livebench.ai" target="_blank" rel="noopener noreferrer" className="text-accent-400 hover:underline">LiveBench</a> benchmarks.
                </p>
                
                {LIVEBENCH_DIMENSIONS.map(({ key, label, description }) => {
                  const value = capabilityExpectations[key] ?? 0
                  const isActive = value > 0
                  
                  return (
                    <div key={key} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <label className={`text-sm font-medium ${isActive ? 'text-void-100' : 'text-void-400'}`}>
                            {label}
                          </label>
                          <p className="text-xs text-void-500">{description}</p>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className={`text-sm font-mono w-8 text-right ${isActive ? 'text-accent-400' : 'text-void-500'}`}>
                            {value > 0 ? value : '-'}
                          </span>
                          {isActive && (
                            <button
                              type="button"
                              onClick={() => updateCapability(key, null)}
                              className="text-void-500 hover:text-void-300 text-xs"
                            >
                              Clear
                            </button>
                          )}
                        </div>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={value}
                        onChange={(e) => {
                          const newValue = parseInt(e.target.value)
                          updateCapability(key, newValue > 0 ? newValue : null)
                        }}
                        className={`w-full h-2 rounded-lg appearance-none cursor-pointer ${
                          isActive 
                            ? 'bg-accent-500/30 accent-accent-500' 
                            : 'bg-void-700 accent-void-500'
                        }`}
                      />
                    </div>
                  )
                })}
                
                {hasCapabilities && (
                  <div className="mt-4 pt-4 border-t border-void-700">
                    <p className="text-sm text-emerald-400">
                      Models will be filtered to meet your minimum scores in: {
                        Object.entries(capabilityExpectations)
                          .filter(([_, v]) => v !== null && v !== undefined && v > 0)
                          .map(([k, v]) => `${LIVEBENCH_DIMENSIONS.find(d => d.key === k)?.label} (≥${v})`)
                          .join(', ')
                      }
                    </p>
                  </div>
                )}
              </div>
            )}
            
            {!showCapabilities && hasCapabilities && (
              <div className="text-sm text-emerald-400 mt-2">
                ✓ {Object.values(capabilityExpectations).filter(v => v !== null && v !== undefined && v > 0).length} capability expectations configured
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between gap-3 pt-4 border-t border-void-700">
            <div className="text-sm text-void-400">
              {formData.selected_log_ids.length > 0 && (
                <span className="text-emerald-400">
                  ✓ {formData.selected_log_ids.length} log{formData.selected_log_ids.length !== 1 ? 's' : ''} will be imported
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <Button 
                type="button" 
                variant="ghost"
                onClick={() => navigate(-1)}
              >
                Cancel
              </Button>
              <Button 
                type="submit" 
                variant="primary"
                loading={createMutation.isPending}
              >
                {createMutation.isPending && formData.selected_log_ids.length > 0
                  ? 'Creating & Importing Logs...'
                  : 'Create Project'}
              </Button>
            </div>
          </div>

          {createMutation.isError && (
            <div className="p-4 bg-accent-500/10 border border-accent-500/30 rounded-lg text-accent-400 text-sm">
              Failed to create project. Please try again.
            </div>
          )}
        </form>
      </Card>
    </div>
  )
}
