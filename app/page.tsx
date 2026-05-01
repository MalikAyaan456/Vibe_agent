"use client";

import { useState } from "react";
import { askGemini } from "@/lib/gemini";

export default function Home() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const handleAsk = async () => {
    setOutput("Thinking...");
    const res = await askGemini(input);
    setOutput(res);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Vibe Agent AI Test</h2>

      <textarea
        rows={4}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        style={{ width: "100%" }}
      />

      <button onClick={handleAsk} style={{ marginTop: 10 }}>
        Ask AI
      </button>

      <pre style={{ marginTop: 20 }}>{output}</pre>
    </div>
  );
}
