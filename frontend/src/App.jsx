
import { useEffect, useState } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [prompt, setPrompt] = useState(
    'You are a sentiment classifier. Read the review and output exactly POSITIVE or NEGATIVE. Never output any other word or explanation.\nReview: {{input}}'
  )
  const [objective, setObjective] = useState('balanced')
  const [minAccuracy, setMinAccuracy] = useState(70)
  const [experiments, setExperiments] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [testResult, setTestResult] = useState(null)
  const [testing, setTesting] = useState(false)
  const [testError, setTestError] = useState('')
  const [testRuns, setTestRuns] = useState([])
  const [selectedTestRun, setSelectedTestRun] = useState(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [activeTab, setActiveTab] = useState('agent')
  const [theme, setTheme] = useState(() => localStorage.getItem('promptlab_theme') || 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('promptlab_theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  const [agentPrompt, setAgentPrompt] = useState(
    'What is the feeling in this text? {{input}}'
  )
  const [agentTarget, setAgentTarget] = useState(90)
  const [agentMaxIter, setAgentMaxIter] = useState(3)
  const [agentRunning, setAgentRunning] = useState(false)
  const [agentError, setAgentError] = useState('')
  const [agentResult, setAgentResult] = useState(null)
  const [agentRuns, setAgentRuns] = useState([])
  const [agentRunsLoading, setAgentRunsLoading] = useState(false)
  const [selectedAgentRun, setSelectedAgentRun] = useState(null)

  async function loadExperiments() {
    try {
      const response = await fetch(`${API}/api/experiments`)
      if (!response.ok) throw new Error('Could not load experiment history')
      const data = await response.json()
      setExperiments(data.experiments || [])
    } catch (err) { setError(`Backend connection failed: ${err.message}`) }
  }
  useEffect(() => { loadExperiments() }, [])

  async function loadTestHistory() {
    setHistoryLoading(true); setHistoryError('')
    try {
      const response = await fetch(`${API}/api/test-runs`)
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Failed to load test history.')
      setTestRuns(data.test_runs || [])
    } catch (err) { setHistoryError(err.message) }
    finally { setHistoryLoading(false) }
  }
  async function openTestRun(id) {
    setHistoryError('')
    try {
      const response = await fetch(`${API}/api/test-runs/${id}`)
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Failed to load test run.')
      setSelectedTestRun(data)
    } catch (err) { setHistoryError(err.message) }
  }
  useEffect(() => { loadTestHistory() }, [])

  async function optimizePrompt(event) {
    event.preventDefault()
    setLoading(true); setError(''); setNotice(''); setSelected(null)
    try {
      const response = await fetch(`${API}/api/optimize`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, objective, min_accuracy: Number(minAccuracy) }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Optimization failed')
      setNotice(`Experiment #${data.experiment_id} completed and saved.`)
      await loadExperiments()
      const detail = await fetch(`${API}/api/experiments/${data.experiment_id}`)
      if (detail.ok) setSelected(await detail.json())
    } catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  async function openExperiment(id) {
    setError('')
    try {
      const response = await fetch(`${API}/api/experiments/${id}`)
      if (!response.ok) throw new Error('Could not open this experiment')
      setSelected(await response.json())
    } catch (err) { setError(err.message) }
  }

  async function testBestPrompt() {
    if (!selected?.best_prompt) { setTestError('Select an experiment with a saved best prompt first.'); return }
    setTesting(true); setTestError(''); setTestResult(null)
    try {
      const response = await fetch(`${API}/api/test?experiment_id=${selected.id}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: selected.best_prompt }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data))
      setTestResult(data); await loadTestHistory()
    } catch (err) { setTestError(err.message) }
    finally { setTesting(false) }
  }

  async function loadAgentRuns() {
    setAgentRunsLoading(true)
    try {
      const response = await fetch(`${API}/api/agent-runs`)
      const data = await response.json()
      setAgentRuns(data.agent_runs || [])
    } catch {} finally { setAgentRunsLoading(false) }
  }
  async function openAgentRun(id) {
    try {
      const response = await fetch(`${API}/api/agent-runs/${id}`)
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail)
      setSelectedAgentRun(data); setAgentResult(null)
    } catch (err) { setAgentError(err.message) }
  }
  useEffect(() => { loadAgentRuns() }, [])

  async function startAgent(event) {
    event.preventDefault()
    setAgentRunning(true); setAgentError(''); setAgentResult(null); setSelectedAgentRun(null)
    try {
      const response = await fetch(`${API}/api/agent/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: agentPrompt, target_accuracy: Number(agentTarget), max_iterations: Number(agentMaxIter) }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Agent run failed')
      setAgentResult(data); await loadAgentRuns()
    } catch (err) { setAgentError(err.message) }
    finally { setAgentRunning(false) }
  }

  const results = selected?.results || []

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">P</div>
          <div>
            <div className="brand-title-wrap">
              <h1>PromptLab</h1>
              <span className="brand-tag">Agentic AI v2.0</span>
            </div>
            <p>Autonomous AI Prompt Testing &amp; Optimization</p>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="theme-toggle" onClick={toggleTheme} title="Toggle Dark/Light Mode">
            <span className="theme-icon">{theme === 'dark' ? '☀️' : '🌙'}</span>
            <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
          <div className="status"><span className="status-dot" />Local AI · Ready</div>
        </div>
      </header>

      <section className="welcome">
        <div>
          <p className="eyebrow">AUTONOMOUS PROMPT OPTIMIZATION</p>
          <h2>Make every prompt measurable.</h2>
          <p className="welcome-copy">Autonomous agentic AI optimization using the ReAct loop (Think → Act → Observe), comparing candidates, and saving verifiable results in SQLite.</p>
        </div>
        <div className="welcome-stat">
          <span>AGENT RUNS RECORDED</span>
          <strong>{agentRuns.length}</strong>
          <small>Stored locally in SQLite</small>
        </div>
      </section>

      <div className="tab-bar">
        <button id="tab-agent" className={`tab-btn ${activeTab === 'agent' ? 'tab-active' : ''}`} onClick={() => setActiveTab('agent')}>
          <span>🤖</span>
          <span>Agentic AI (Autonomous)</span>
          <span className="tab-badge">{agentRuns.length}</span>
        </button>
        <button id="tab-lab" className={`tab-btn ${activeTab === 'lab' ? 'tab-active' : ''}`} onClick={() => setActiveTab('lab')}>
          <span>🧪</span>
          <span>Prompt Lab</span>
          <span className="tab-badge">{experiments.length}</span>
        </button>
      </div>

      {activeTab === 'lab' && (
        <>
          {error && <div className="alert error">{error}</div>}
          {notice && <div className="alert success">{notice}</div>}
          <div className="workspace">
            <section className="panel editor-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">01 / CONFIGURE</p><h3>Prompt experiment</h3></div>
                <span className="pill">llama3.2:1b</span>
              </div>
              <form onSubmit={optimizePrompt}>
                <label htmlFor="prompt">Baseline prompt</label>
                <textarea id="prompt" value={prompt} onChange={e => setPrompt(e.target.value)} rows={7} required />
                <label htmlFor="objective">Optimization objective</label>
                <select id="objective" value={objective} onChange={e => setObjective(e.target.value)}>
                  <option value="balanced">Balanced — accuracy and speed</option>
                  <option value="accuracy">Accuracy first</option>
                  <option value="speed">Speed first</option>
                </select>
                <div className="range-heading">
                  <label htmlFor="accuracy">Minimum accuracy target</label>
                  <strong>{minAccuracy}%</strong>
                </div>
                <input id="accuracy" type="range" min="0" max="100" step="10" value={minAccuracy} onChange={e => setMinAccuracy(e.target.value)} />
                <button className="primary-button" type="submit" disabled={loading}>
                  {loading ? 'Running experiment…' : 'Run optimization'}{!loading && <span>↗</span>}
                </button>
                <p className="form-note">Runs locally. First-time experiments may take longer.</p>
              </form>
            </section>
            <section className="panel history-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">02 / HISTORY</p><h3>Saved experiments</h3></div>
                <button className="text-button" onClick={loadExperiments}>Refresh ↻</button>
              </div>
              {experiments.length === 0 ? (
                <div className="empty-state"><div className="empty-icon">⌁</div><h4>No experiments yet</h4><p>Run an optimization to save your first result here.</p></div>
              ) : (
                <div className="history-list">
                  {experiments.map(experiment => (
                    <button className={`history-item ${selected?.id === experiment.id ? 'active' : ''}`} key={experiment.id} onClick={() => openExperiment(experiment.id)}>
                      <div className="history-top"><strong>Experiment #{experiment.id}</strong><span className="history-accuracy">{experiment.accuracy_percent}%</span></div>
                      <p>{experiment.objective} · {experiment.model}</p>
                      <div className="history-bottom"><span>{experiment.dataset_size} examples</span><span>{Number(experiment.average_latency_ms).toFixed(0)} ms</span></div>
                    </button>
                  ))}
                </div>
              )}
            </section>
          </div>
          {selected && (
            <section className="panel results-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">03 / RESULTS</p><h3>Experiment #{selected.id}</h3><p className="subheading">{selected.created_at} · {selected.model} · {selected.objective}</p></div>
                <button className="text-button" onClick={() => setSelected(null)}>Close ×</button>
              </div>
              <div className="metrics-grid">
                <Metric label="Best accuracy" value={`${selected.accuracy_percent}%`} />
                <Metric label="Average latency" value={`${Number(selected.average_latency_ms).toFixed(0)} ms`} />
                <Metric label="Invalid outputs" value={selected.invalid_outputs} />
                <Metric label="Dataset size" value={selected.dataset_size} />
              </div>
              <h4 className="section-label">Candidate comparison</h4>
              {results.length === 0 ? <p>No candidate details were returned for this experiment.</p> : (
                <div className="table-wrap"><table>
                  <thead><tr><th>Candidate</th><th>Accuracy</th><th>Latency</th><th>Invalid</th></tr></thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr key={result.name || index}>
                        <td><strong>{result.name || `Candidate ${index + 1}`}</strong>{result.name === selected.best_candidate_name && <span className="best-tag">BEST</span>}</td>
                        <td>{result.accuracy ?? result.accuracy_percent ?? result.metrics?.accuracy ?? '—'}{(result.accuracy ?? result.accuracy_percent ?? result.metrics?.accuracy) != null ? '%' : ''}</td>
                        <td>{Number(result.average_latency_ms).toFixed(0)} ms</td>
                        <td>{result.invalid_outputs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table></div>
              )}
              <div className="test-actions">
                <div><h4 className="section-label">Held-out test evaluation</h4><p className="test-description">Evaluate the saved best prompt on separate test examples.</p></div>
                <button className="primary-button test-button" onClick={testBestPrompt} disabled={testing}>{testing ? 'Testing prompt…' : 'Test best prompt'}</button>
              </div>
              {testError && <div className="alert error">{testError}</div>}
              {testResult && (
                <div className="test-result">
                  <div className="test-result-heading">
                    <div><h4>Held-out test results</h4><p className="test-description">{testResult.dataset_name} · {testResult.model}</p></div>
                    <span className="test-score">{testResult.result.accuracy_percent}% accuracy</span>
                  </div>
                  <div className="metrics-grid test-metrics">
                    <Metric label="Correct predictions" value={`${testResult.result.correct}/${testResult.result.total_examples}`} />
                    <Metric label="Average latency" value={`${Number(testResult.result.average_latency_ms).toFixed(0)} ms`} />
                    <Metric label="Invalid outputs" value={testResult.result.invalid_outputs} />
                    <Metric label="Avg. output tokens" value={testResult.result.average_output_tokens} />
                  </div>
                  <h4 className="section-label">Prediction breakdown</h4>
                  <div className="table-wrap"><table>
                    <thead><tr><th>Review</th><th>Expected</th><th>Predicted</th><th>Latency</th><th>Result</th></tr></thead>
                    <tbody>
                      {testResult.result.details.map((item, index) => (
                        <tr key={index}>
                          <td className="review-cell">{item.text}</td>
                          <td>{item.expected}</td><td>{item.predicted}</td>
                          <td>{Number(item.latency_ms).toFixed(0)} ms</td>
                          <td><span className={item.correct ? 'pass-tag' : 'fail-tag'}>{item.correct ? 'Correct' : 'Incorrect'}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table></div>
                  <p className="test-warning">{testResult.warning}</p>
                </div>
              )}
              <h4 className="section-label">Best prompt</h4>
              <pre className="prompt-output">{selected.best_prompt}</pre>
            </section>
          )}
          <section className="panel test-history-panel">
            <div className="panel-heading">
              <div><p className="eyebrow">04 / TEST HISTORY</p><h3>Held-out test runs</h3><p className="subheading">Review saved evaluations and inspect individual predictions.</p></div>
              <button className="text-button" onClick={loadTestHistory} disabled={historyLoading}>{historyLoading ? 'Loading…' : 'Refresh ↻'}</button>
            </div>
            {historyError && <div className="alert error">{historyError}</div>}
            {historyLoading && testRuns.length === 0 ? <p>Loading test history…</p>
              : testRuns.length === 0 ? <div className="empty-state"><h4>No test runs yet</h4><p>Test an experiment's best prompt to save an evaluation.</p></div>
              : (
                <div className="table-wrap"><table>
                  <thead><tr><th>Run</th><th>Experiment</th><th>Accuracy</th><th>Correct</th><th>Latency</th><th>Invalid</th><th>Details</th></tr></thead>
                  <tbody>
                    {testRuns.map(run => (
                      <tr key={run.id}>
                        <td><strong>#{run.id}</strong><br /><small>{run.created_at}</small></td>
                        <td>{run.experiment_id != null ? `#${run.experiment_id}` : 'Unlinked'}</td>
                        <td><strong>{run.accuracy_percent}%</strong></td>
                        <td>{run.correct}/{run.total_examples}</td>
                        <td>{Number(run.average_latency_ms).toFixed(0)} ms</td>
                        <td>{run.invalid_outputs}</td>
                        <td><button className="text-button" onClick={() => openTestRun(run.id)}>View details</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table></div>
              )
            }
            {selectedTestRun && (
              <div className="test-result">
                <div className="test-result-heading">
                  <div><h4>Test Run #{selectedTestRun.id}</h4><p className="test-description">{selectedTestRun.dataset_name} · {selectedTestRun.model}</p></div>
                  <button className="text-button" onClick={() => setSelectedTestRun(null)}>Close ×</button>
                </div>
                <div className="metrics-grid test-metrics">
                  <Metric label="Accuracy" value={`${selectedTestRun.accuracy_percent}%`} />
                  <Metric label="Correct predictions" value={`${selectedTestRun.correct}/${selectedTestRun.total_examples}`} />
                  <Metric label="Average latency" value={`${Number(selectedTestRun.average_latency_ms).toFixed(0)} ms`} />
                  <Metric label="Invalid outputs" value={selectedTestRun.invalid_outputs} />
                </div>
                <h4 className="section-label">Individual predictions</h4>
                <div className="table-wrap"><table>
                  <thead><tr><th>Review</th><th>Expected</th><th>Predicted</th><th>Raw response</th><th>Latency</th><th>Result</th></tr></thead>
                  <tbody>
                    {(selectedTestRun.result?.result?.details || []).map((item, index) => (
                      <tr key={index}>
                        <td className="review-cell">{item.text}</td>
                        <td>{item.expected}</td><td>{item.predicted ?? 'Invalid'}</td>
                        <td>{item.response}</td>
                        <td>{Number(item.latency_ms).toFixed(0)} ms</td>
                        <td><span className={item.correct ? 'pass-tag' : 'fail-tag'}>{item.correct ? 'Correct' : 'Incorrect'}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table></div>
                <h4 className="section-label">Prompt used for this run</h4>
                <pre className="prompt-output">{selectedTestRun.prompt}</pre>
              </div>
            )}
          </section>
        </>
      )}

      {activeTab === 'agent' && (
        <>
          <div className="agent-banner">
            <div className="agent-banner-icon">🤖</div>
            <div>
              <h3>Autonomous Prompt Optimization Agent</h3>
              <p>Set a target accuracy goal and let the agent work autonomously. It uses a <strong>ReAct loop</strong> (Think → Act → Observe) to evaluate prompts, analyse failures, generate improved candidates, and iterate — all without you clicking anything.</p>
            </div>
          </div>
          {agentError && <div className="alert error">{agentError}</div>}
          <div className="workspace">
            <section className="panel editor-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">AGENT / CONFIGURE</p><h3>Set your goal</h3></div>
                <span className="pill agent-pill">ReAct Agent</span>
              </div>
              <form onSubmit={startAgent}>
                <label htmlFor="agent-prompt">Starting prompt</label>
                <textarea id="agent-prompt" value={agentPrompt} onChange={e => setAgentPrompt(e.target.value)} rows={6} required />
                <div className="range-heading"><label htmlFor="agent-target">Target accuracy</label><strong>{agentTarget}%</strong></div>
                <input id="agent-target" type="range" min="10" max="100" step="10" value={agentTarget} onChange={e => setAgentTarget(e.target.value)} />
                <div className="range-heading"><label htmlFor="agent-iter">Max iterations</label><strong>{agentMaxIter}</strong></div>
                <input id="agent-iter" type="range" min="1" max="5" step="1" value={agentMaxIter} onChange={e => setAgentMaxIter(e.target.value)} />
                <button id="start-agent-btn" className="primary-button agent-start-btn" type="submit" disabled={agentRunning}>
                  {agentRunning ? <><span className="spinner" />Agent is running…</> : <>🚀 Start Agent</>}
                </button>
                {agentRunning && <p className="form-note agent-running-note">⏳ Agent is autonomously thinking, evaluating, and improving the prompt. This may take several minutes. Please wait…</p>}
              </form>
            </section>
            <section className="panel history-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">AGENT / HISTORY</p><h3>Past agent runs</h3></div>
                <button className="text-button" onClick={loadAgentRuns}>Refresh ↻</button>
              </div>
              {agentRunsLoading ? <p>Loading…</p> : agentRuns.length === 0 ? (
                <div className="empty-state"><div className="empty-icon">🤖</div><h4>No agent runs yet</h4><p>Start an agent run to see history here.</p></div>
              ) : (
                <div className="history-list">
                  {agentRuns.map(run => (
                    <button key={run.id} className={`history-item ${selectedAgentRun?.id === run.id ? 'active' : ''}`} onClick={() => openAgentRun(run.id)}>
                      <div className="history-top">
                        <strong>Agent Run #{run.id}</strong>
                        <span className={`history-accuracy ${run.goal_achieved ? 'goal-met' : 'goal-missed'}`}>{run.best_accuracy}%</span>
                      </div>
                      <p>Target: {run.target_accuracy}% · {run.total_iterations} iteration(s)</p>
                      <div className="history-bottom">
                        <span>{run.goal_achieved ? '✅ Goal met' : '❌ Not met'}</span>
                        <span>{run.created_at?.slice(0, 10)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </section>
          </div>
          {agentResult && (
            <section className="panel agent-result-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">AGENT / RESULT</p><h3>Run #{agentResult.agent_run_id} Complete</h3></div>
                <span className={`pill ${agentResult.goal_achieved ? 'pill-success' : 'pill-warn'}`}>{agentResult.goal_achieved ? '✅ Goal Achieved' : '⚠️ Goal Not Met'}</span>
              </div>
              <div className="metrics-grid">
                <Metric label="Best accuracy found" value={`${agentResult.best_accuracy}%`} />
                <Metric label="Target accuracy" value={`${agentResult.target_accuracy}%`} />
                <Metric label="Iterations run" value={agentResult.total_iterations} />
                <Metric label="Goal achieved?" value={agentResult.goal_achieved ? 'Yes ✅' : 'No ❌'} />
              </div>
              <h4 className="section-label">🧠 Agent Reasoning Trace (Think → Act → Observe)</h4>
              <div className="trace-log">
                {agentResult.trace.map((step, i) => (
                  <div key={i} className={`trace-step trace-${step.type.toLowerCase()}`}>
                    <span className="trace-badge">{step.type}</span>
                    <span className="trace-iter">Iter {step.iteration}</span>
                    <span className="trace-msg">{step.message}</span>
                  </div>
                ))}
              </div>
              <h4 className="section-label">🏆 Best Prompt Found by Agent</h4>
              <pre className="prompt-output">{agentResult.best_prompt}</pre>
            </section>
          )}
          {selectedAgentRun && !agentResult && (
            <section className="panel agent-result-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">AGENT / HISTORY DETAIL</p><h3>Agent Run #{selectedAgentRun.id}</h3><p className="subheading">{selectedAgentRun.created_at} · {selectedAgentRun.model}</p></div>
                <button className="text-button" onClick={() => setSelectedAgentRun(null)}>Close ×</button>
              </div>
              <div className="metrics-grid">
                <Metric label="Best accuracy" value={`${selectedAgentRun.best_accuracy}%`} />
                <Metric label="Target accuracy" value={`${selectedAgentRun.target_accuracy}%`} />
                <Metric label="Iterations" value={selectedAgentRun.total_iterations} />
                <Metric label="Goal achieved?" value={selectedAgentRun.goal_achieved ? 'Yes ✅' : 'No ❌'} />
              </div>
              <h4 className="section-label">🧠 Agent Reasoning Trace</h4>
              <div className="trace-log">
                {(selectedAgentRun.trace_log || []).map((step, i) => (
                  <div key={i} className={`trace-step trace-${step.type.toLowerCase()}`}>
                    <span className="trace-badge">{step.type}</span>
                    <span className="trace-iter">Iter {step.iteration}</span>
                    <span className="trace-msg">{step.message}</span>
                  </div>
                ))}
              </div>
              <h4 className="section-label">🏆 Best Prompt Found</h4>
              <pre className="prompt-output">{selectedAgentRun.best_prompt}</pre>
            </section>
          )}
        </>
      )}

      <footer>
        <span>PromptLab</span>
        <span>Local-first · Experimental results are dataset-dependent</span>
      </footer>
    </main>
  )
}

function Metric({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value ?? '—'}</strong>
    </div>
  )
}

export default App
