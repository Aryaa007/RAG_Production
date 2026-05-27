import { useState, useRef, useEffect } from "react"
import axios from "axios"


// Three sections: upload bar, chat window, input bar

export default function App() {

  // useState is like a variable that re-renders the UI when it changes

  const [messages,   setMessages]   = useState([])        // chat history
  const [question,   setQuestion]   = useState("")        // current input text
  const [loading,    setLoading]    = useState(false)     // true while waiting for answer
  const [uploading,  setUploading]  = useState(false)     // true while uploading PDF
  const [activeFile, setActiveFile] = useState(null)      // currently loaded PDF name
  const [error,      setError]      = useState(null)      // error message if any

  // ref to auto-scroll chat to bottom on new messages
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // ── upload handler ────────────────────────────────────────────────────────────
  // called when user picks a file
  // sends it to POST /upload on your FastAPI backend

  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file) return

    setUploading(true)
    setError(null)

    // FormData is how you send files in HTTP requests
    const formData = new FormData()
    formData.append("file", file)   // "file" must match the FastAPI parameter name

    try {
      const res = await axios.post("http://localhost:8000/upload", formData)
      setActiveFile(res.data.filename)

      // add a system message to chat
      setMessages(prev => [...prev, {
        role:    "system",
        content: `📄 Loaded: ${res.data.filename}`
      }])
    } catch (_err) {
      setError("Upload failed. Make sure the backend is running.")
    } finally {
      setUploading(false)
    }
  }

  // ── ask handler ───────────────────────────────────────────────────────────────
  // called when user hits Send or presses Enter
  // sends question to POST /ask on your FastAPI backend

  async function handleAsk() {
    if (!question.trim() || loading) return

    const userQuestion = question
    setQuestion("")   // clear input immediately

    // add user message to chat
    setMessages(prev => [...prev, {
      role:    "user",
      content: userQuestion
    }])

    setLoading(true)
    setError(null)

    try {
      const res = await axios.post("http://localhost:8000/ask", {
        question: userQuestion   // matches QuestionRequest model in FastAPI
      })

      // add AI answer to chat
      setMessages(prev => [...prev, {
        role:    "assistant",
        content: res.data.answer,
        sources: res.data.sources
      }])
    } catch (_err) {
      setError("Something went wrong. Check the backend terminal.")
    } finally {
      setLoading(false)
    }
  }

  // Enter key sends message
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  // ── render ────────────────────────────────────────────────────────────────────

  return (
    <div style={styles.app}>

      {/* ── header ── */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.logo}>⬡</span>
          <span style={styles.title}>DocMind</span>
        </div>
        <div style={styles.uploadArea}>
          {activeFile && (
            <span style={styles.activeFile}>
              📄 {activeFile}
            </span>
          )}
          <label style={styles.uploadBtn}>
            {uploading ? "Uploading..." : "Upload PDF"}
            <input
              type="file"
              accept=".pdf"
              onChange={handleUpload}
              style={{ display: "none" }}
            />
          </label>
        </div>
      </div>

      {/* ── chat window ── */}
      <div style={styles.chatWindow}>

        {/* empty state */}
        {messages.length === 0 && (
          <div style={styles.emptyState}>
            <div style={styles.emptyIcon}>⬡</div>
            <p style={styles.emptyTitle}>Upload a PDF to get started</p>
            <p style={styles.emptySubtitle}>
              Ask anything about your document — answers come with citations
            </p>
          </div>
        )}

        {/* messages */}
        {messages.map((msg, i) => (
          <div key={i} style={{
            ...styles.messageRow,
            justifyContent: msg.role === "user" ? "flex-end" : "flex-start"
          }}>

            {/* system message — file loaded notification */}
            {msg.role === "system" && (
              <div style={styles.systemMsg}>{msg.content}</div>
            )}

            {/* user message */}
            {msg.role === "user" && (
              <div style={styles.userBubble}>{msg.content}</div>
            )}

            {/* assistant message */}
            {msg.role === "assistant" && (
              <div style={styles.aiBubble}>
                <p style={styles.aiText}>{msg.content}</p>
                {msg.sources && msg.sources.length > 0 && (
                  <div style={styles.sources}>
                    <span style={styles.sourcesLabel}>Sources: </span>
                    {msg.sources.map((s, j) => (
                      <span key={j} style={styles.sourceTag}>{s}</span>
                    ))}
                  </div>
                )}
              </div>
            )}

          </div>
        ))}

        {/* loading indicator */}
        {loading && (
          <div style={styles.messageRow}>
            <div style={styles.aiBubble}>
              <div style={styles.typingDots}>
                <span style={{...styles.dot, animationDelay: "0ms"}}>●</span>
                <span style={{...styles.dot, animationDelay: "200ms"}}>●</span>
                <span style={{...styles.dot, animationDelay: "400ms"}}>●</span>
              </div>
            </div>
          </div>
        )}

        {/* error message */}
        {error && <div style={styles.errorMsg}>{error}</div>}

        {/* scroll anchor */}
        <div ref={bottomRef} />

      </div>

      {/* ── input bar ── */}
      <div style={styles.inputBar}>
        <textarea
          style={styles.input}
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your document..."
          rows={1}
        />
        <button
          style={{
            ...styles.sendBtn,
            opacity: loading || !question.trim() ? 0.4 : 1
          }}
          onClick={handleAsk}
          disabled={loading || !question.trim()}
        >
          Send
        </button>
      </div>

    </div>
  )
}

// ── styles ─────────────────────────────────────────────────────────────────────
// Inline styles — no CSS file needed, everything in one place

const styles = {
  app: {
    display:         "flex",
    flexDirection:   "column",
    height:          "100vh",
    backgroundColor: "#0a0a0f",
    color:           "#e8e6e1",
    fontFamily:      "'IBM Plex Mono', monospace",
  },

  // header
  header: {
    display:         "flex",
    justifyContent:  "space-between",
    alignItems:      "center",
    padding:         "16px 24px",
    borderBottom:    "1px solid #1e1e2e",
    backgroundColor: "#0d0d16",
  },
  headerLeft: {
    display:    "flex",
    alignItems: "center",
    gap:        "10px",
  },
  logo: {
    fontSize: "22px",
    color:    "#7c6af7",
  },
  title: {
    fontSize:   "18px",
    fontWeight: "600",
    color:      "#e8e6e1",
    letterSpacing: "0.05em",
  },
  uploadArea: {
    display:    "flex",
    alignItems: "center",
    gap:        "12px",
  },
  activeFile: {
    fontSize:        "12px",
    color:           "#7c6af7",
    backgroundColor: "#1a1a2e",
    padding:         "4px 10px",
    borderRadius:    "4px",
    border:          "1px solid #2a2a4a",
  },
  uploadBtn: {
    backgroundColor: "#7c6af7",
    color:           "#fff",
    padding:         "8px 16px",
    borderRadius:    "6px",
    cursor:          "pointer",
    fontSize:        "13px",
    fontWeight:      "500",
    fontFamily:      "'IBM Plex Mono', monospace",
  },

  // chat
  chatWindow: {
    flex:      1,
    overflowY: "auto",
    padding:   "24px",
    display:   "flex",
    flexDirection: "column",
    gap:       "16px",
  },
  emptyState: {
    margin:    "auto",
    textAlign: "center",
    opacity:   0.4,
  },
  emptyIcon: {
    fontSize:     "48px",
    color:        "#7c6af7",
    marginBottom: "16px",
  },
  emptyTitle: {
    fontSize:   "18px",
    fontWeight: "600",
    margin:     "0 0 8px 0",
  },
  emptySubtitle: {
    fontSize: "14px",
    color:    "#888",
    margin:   0,
  },
  messageRow: {
    display: "flex",
    width:   "100%",
  },
  systemMsg: {
    margin:          "0 auto",
    fontSize:        "12px",
    color:           "#7c6af7",
    backgroundColor: "#1a1a2e",
    padding:         "6px 14px",
    borderRadius:    "20px",
    border:          "1px solid #2a2a4a",
  },
  userBubble: {
    backgroundColor: "#7c6af7",
    color:           "#fff",
    padding:         "12px 16px",
    borderRadius:    "16px 16px 4px 16px",
    maxWidth:        "70%",
    fontSize:        "14px",
    lineHeight:      "1.6",
  },
  aiBubble: {
    backgroundColor: "#13131f",
    border:          "1px solid #1e1e2e",
    padding:         "14px 18px",
    borderRadius:    "16px 16px 16px 4px",
    maxWidth:        "75%",
    fontSize:        "14px",
    lineHeight:      "1.7",
  },
  aiText: {
    margin: "0 0 10px 0",
  },
  sources: {
    display:    "flex",
    flexWrap:   "wrap",
    gap:        "6px",
    alignItems: "center",
    marginTop:  "8px",
    paddingTop: "8px",
    borderTop:  "1px solid #1e1e2e",
  },
  sourcesLabel: {
    fontSize: "11px",
    color:    "#666",
  },
  sourceTag: {
    fontSize:        "11px",
    color:           "#7c6af7",
    backgroundColor: "#1a1a2e",
    padding:         "2px 8px",
    borderRadius:    "4px",
    border:          "1px solid #2a2a4a",
  },
  typingDots: {
    display: "flex",
    gap:     "6px",
  },
  dot: {
    fontSize:        "8px",
    color:           "#7c6af7",
    animation:       "pulse 1s infinite",
    display:         "inline-block",
  },
  errorMsg: {
    color:           "#ff6b6b",
    backgroundColor: "#1a0f0f",
    border:          "1px solid #3a1a1a",
    padding:         "10px 16px",
    borderRadius:    "8px",
    fontSize:        "13px",
    textAlign:       "center",
  },

  // input
  inputBar: {
    display:         "flex",
    gap:             "12px",
    padding:         "16px 24px",
    borderTop:       "1px solid #1e1e2e",
    backgroundColor: "#0d0d16",
  },
  input: {
    flex:            1,
    backgroundColor: "#13131f",
    border:          "1px solid #1e1e2e",
    borderRadius:    "8px",
    color:           "#e8e6e1",
    padding:         "12px 16px",
    fontSize:        "14px",
    fontFamily:      "'IBM Plex Mono', monospace",
    resize:          "none",
    outline:         "none",
  },
  sendBtn: {
    backgroundColor: "#7c6af7",
    color:           "#fff",
    border:          "none",
    borderRadius:    "8px",
    padding:         "0 24px",
    fontSize:        "14px",
    fontWeight:      "600",
    cursor:          "pointer",
    fontFamily:      "'IBM Plex Mono', monospace",
  },
}