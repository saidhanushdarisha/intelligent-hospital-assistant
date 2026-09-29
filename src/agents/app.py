import uuid

from orchestrator.graph import graph


def save_graph_diagram():
    """Best-effort: draws the graph structure to langgraph.png.
    Wrapped in try/except since draw_mermaid_png() calls an external
    API and shouldn't ever break the chat if there's no internet."""
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("langgraph.png", "wb") as f:
            f.write(png_bytes)
    except Exception as exc:
        print(f"(Skipped saving graph diagram: {exc})")


def run_chat():
    # 1. Make sure the database exists and has baseline data,
    #    every time the app starts.

    save_graph_diagram()

    # 2. One thread_id per session = one continuous conversation.
    #    The SqliteSaver checkpointer in graph.py uses this to load
    #    prior state and save new state after every turn.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Tracks how many messages we've already printed, since "msg" in
    # OrchestratorState accumulates across the WHOLE conversation
    # (it uses an operator.add reducer) rather than resetting each turn.
    printed_count = 0

    print("Hospital Assistant — type 'exit' or 'quit' to end the chat.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break

        result = graph.invoke(
            {
                "user_input": user_input,
                "route": [],
                "msg": [],
            },
            config=config,
        )

        new_messages = result["msg"][printed_count:]
        printed_count = len(result["msg"])

        for line in new_messages:
            print(f"Assistant: {line}")
        print()


if __name__ == "__main__":
    run_chat()