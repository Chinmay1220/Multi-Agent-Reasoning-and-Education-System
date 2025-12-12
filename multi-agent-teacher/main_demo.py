from src.llm_client import LLMClient
from src.orchestration.pipeline_full import MultiAgentPipeline


def main():
    client = LLMClient()
    pipeline = MultiAgentPipeline(client)

    print("=== Multi-Agent Teaching Demo ===")
    question = input("Enter your question about multi-agent LLM systems:\n> ")

    result = pipeline.answer_question(question)

    print("\n--- PLAN ---")
    print(result["plan"])

    print("\n--- FINAL ANSWER ---")
    print(result["final_answer"])

    print("\n--- CRITIQUE (for teaching) ---")
    print(result["critique"])

    print("\n(You can inspect intermediate steps in code or notebooks.)")


if __name__ == "__main__":
    main()
