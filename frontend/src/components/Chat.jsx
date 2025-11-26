// frontend/src/components/Chat.jsx
import React, { useState, useRef, useEffect } from "react";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function Chat() {
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");
  const [lastError, setLastError] = useState(null);
  const chatHistory = useRef([]);
  const containerRef = useRef(null);

  useEffect(() => {
    containerRef.current?.scrollTo({
      top: containerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function upload() {
    if (!file) return alert("Choose a PDF first.");
    setUploading(true);
    setStatus("Uploading...");
    setLastError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) {
        setLastError(data?.detail || data || "Upload failed");
        setStatus("Upload failed");
      } else {
        setStatus(`Uploaded: ${data.filename || "ok"}`);
        setFile(null);
      }
    } catch (err) {
      console.error(err);
      setLastError(String(err));
      setStatus("Upload failed (network)");
    } finally {
      setUploading(false);
    }
  }

  async function send() {
    const question = input.trim();
    if (!question) return;
    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    setLastError(null);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, chat_history: chatHistory.current }),
      });
      const data = await res.json();
      if (!res.ok) {
        setLastError(data || "Server error");
        setMessages((m) => [
          ...m,
          { role: "assistant", content: `Error: ${JSON.stringify(data)}` },
        ]);
      } else {
        const answer = data.answer ?? null;
        if (!answer) {
          // Show full returned payload for debugging
          setMessages((m) => [
            ...m,
            { role: "assistant", content: "No answer returned; server response:\n" + JSON.stringify(data) },
          ]);
        } else {
          setMessages((m) => [...m, { role: "assistant", content: answer }]);
          chatHistory.current.push([question, answer]);
          // Optionally show source previews
          if (data.source_documents && data.source_documents.length) {
            setMessages((m) => [
              ...m,
              {
                role: "assistant",
                content:
                  "Sources:\n" +
                  data.source_documents
                    .map((s, i) => `(${i + 1}) ${JSON.stringify(s).slice(0, 250)}`)
                    .join("\n\n"),
              },
            ]);
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages((m) => [...m, { role: "assistant", content: "Network error. See console." }]);
      setLastError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 18, maxWidth: 980, margin: "8px auto", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ margin: "0 0 12px 0" }}>RagNote — Local RAG (Ollama + MiniLM)</h2>

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button onClick={upload} disabled={!file || uploading}>
          {uploading ? "Uploading..." : "Upload & Index"}
        </button>
        <div style={{ marginLeft: 12, color: "#444" }}>{status}</div>
        {lastError && (
          <div style={{ marginLeft: 12, color: "crimson", fontSize: 13 }}>
            {typeof lastError === "string" ? lastError : JSON.stringify(lastError)}
          </div>
        )}
      </div>

      <div
        ref={containerRef}
        style={{
          minHeight: 320,
          maxHeight: 520,
          overflow: "auto",
          background: "#f6f7f9",
          padding: 12,
          borderRadius: 8,
          border: "1px solid #ddd",
        }}
      >
        {messages.length === 0 && (
          <div style={{ color: "#666" }}>No messages yet. Upload a PDF and ask a question.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 12, display: "flex", justifyContent: m.role === "assistant" ? "flex-start" : "flex-end" }}>
            <div
              style={{
                background: m.role === "assistant" ? "#fff" : "#0ea5a4",
                color: m.role === "assistant" ? "#111" : "#fff",
                padding: 10,
                borderRadius: 8,
                maxWidth: "80%",
                whiteSpace: "pre-wrap",
              }}
            >
              {m.content}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the uploaded PDF..."
          style={{ flex: 1, padding: 10 }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !loading) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button onClick={send} disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </div>

      <div style={{ marginTop: 10, color: "#666", fontSize: 13 }}>
        Backend: {API_BASE}
      </div>
    </div>
  );
}
