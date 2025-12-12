# Teaching Multi-Agent LLM Systems: Planner–Retriever–Analyst–Critic

This project is my take-home final for **INFO 7390: Advanced Data Science and Architecture**.

I use a **multi-agent LLM system** to *teach* how agents collaborate:
- **Planner**: clarifies and decomposes the question
- **RetrieverAgent**: gathers evidence from a small corpus
- **Analyst**: synthesizes a grounded answer
- **Critic**: acts as a computational skeptic and checks for hallucinations

The project includes:
- A working implementation (`src/`)
- Interactive notebooks (`notebooks/`)
- A tutorial document (`tutorial/`)
- Exercises and debugging scenarios (`assessment/`)
- A 10-minute Show-and-Tell video (link below)

---

## 🔍 Concept Overview

Multi-agent LLM systems decompose a complex task into specialized roles.
Instead of one giant prompt, multiple focused agents:
- improve reliability,
- make the reasoning process more interpretable,
- and connect naturally to course themes like **Botspeak**, **GIGO**, and **Computational Skepticism**.

This project demonstrates a **Planner → RetrieverAgent → Analyst → Critic** architecture
for answering questions about a small corpus of teaching documents.

---

## 🎯 Learning Objectives

After working through this material, a learner will be able to:

1. Explain what a multi-agent LLM system is and why you might use one.
2. Implement a simple multi-agent pipeline in Python.
3. Design basic agent prompts for planning, retrieval, analysis, and critique.
4. Diagnose common failure modes like hallucinations and bad retrieval.
5. Extend the system with new agent roles (e.g., Style Editor, Explainer).

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/multi-agent-teacher.git
cd multi-agent-teacher
pip install -r requirements.txt
