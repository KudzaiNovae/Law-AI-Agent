"""CLI for the Zimbabwe law agent.

One-shot:    python cli.py "Can the state acquire agricultural land?"
Interactive: python cli.py
"""
import sys

from agent import LawAgent


def main() -> None:
    agent = LawAgent()
    if len(sys.argv) > 1:
        print(agent.ask(" ".join(sys.argv[1:])))
        return

    print("Zimbabwe Law Agent — ctrl-c or 'quit' to exit")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        print("\n" + agent.ask(q))


if __name__ == "__main__":
    main()
