# 📘 Teaching Multi-Agent LLM Systems: Planner–Retriever–Analyst–Critic

This project is my **take-home final for INFO 7390: Advanced Data Science and Architecture**.  
It is a fully functioning **multi-agent LLM teaching system** designed to show *how multiple AI agents collaborate to answer questions more reliably, transparently, and interpretably than a single LLM*.

The goal of this project is both **technical** (building a modular multi-agent architecture) and **pedagogical** (teaching AI reasoning, Botspeak roles, GIGO, computational skepticism, and retrieval-augmented workflows).

---

# 🚀 Project Overview

This project demonstrates a four-agent architecture:

- **Planner** — Clarifies the user’s question and breaks it into subtasks  
- **RetrieverAgent** — Performs embedding-based retrieval over a small teaching corpus  
- **Analyst** — Synthesizes a grounded answer using retrieved evidence  
- **Critic** — Evaluates the draft answer, identifies hallucinations, and provides a corrected final answer  

Instead of generating a single opaque LLM response, the system makes every step **visible, inspectable, and teachable**.  
This is achieved through:

- A **Streamlit UI** that exposes intermediate agent steps  
- **Teaching notebooks** that walk from single-agent → two-agent → full four-agent systems  
- **A tutorial document** explaining concepts and workflows  
- **Debugging scenarios** to teach failure cases  
- **A Show-and-Tell video** following the Explain → Show → Try method  

This structure creates an **X-ray view of LLM reasoning**, making it ideal for learning, demonstration, and interviews.

---

# 🔍 Concept Overview

Multi-agent LLM systems decompose a complex task into smaller, focused roles.  
Rather than depending on one large prompt, this approach uses **agent specialization**:

- Improves reliability  
- Makes reasoning interpretable  
- Supports grounded evidence-based answers  
- Reduces hallucinations  
- Aligns with INFO 7390 themes like:
  - **Botspeak Framework**
  - **GIGO: Garbage In, Garbage Out**
  - **Computational Skepticism**
  - **Structured prompting**
  - **Retrieval-Augmented Reasoning**

This project implements a clean, modular **Planner → RetrieverAgent → Analyst → Critic** pipeline over a small corpus of instructional text files.

---

# 🎯 Learning Objectives

By exploring this project, a student will be able to:

### Core Understanding
✔ Explain what a multi-agent LLM system is  
✔ Understand why task decomposition improves reliability  
✔ Interpret evidence-backed answer generation  
✔ Recognize how hallucinations arise and how agents reduce them  

### Implementation Skills
✔ Implement a multi-agent pipeline in Python  
✔ Write agent prompts inspired by Botspeak  
✔ Use embeddings for simple retrieval  
✔ Build and evaluate agent reasoning chains  

### Debugging / Extension
✔ Diagnose failure cases  
✔ Explore GIGO effects through retrieval quality  
✔ Modify agent prompts  
✔ Extend the system with new roles (e.g., Style Editor, Explainer, Verifier)  

---

# 🧠 How the System Works — Step-by-Step

Below is the full reasoning pipeline executed for each user question:

---

## 1️⃣ User Question  
The user types a question into the Streamlit app (or CLI).  
Example:  
> “Why might multi-agent LLM systems be more reliable than a single agent?”

---

## 2️⃣ Planner Agent — *Clarify & Decompose*  
The Planner agent:  
- Clarifies the question  
- Rewrites it if necessary  
- Breaks it into smaller, concrete subtasks  

Example output:

```json
{
  "clarified_question": "Explain why using multiple specialized LLM agents can be more reliable than a single agent.",
  "subtasks": [
    "Define what a multi-agent LLM system is.",
    "Identify typical agent roles.",
    "Compare multi-agent workflows to single-agent workflows.",
    "Explain how they reduce hallucinations."
  ]
}
