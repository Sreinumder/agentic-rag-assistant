from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.assistant import AIAssistant
from src.agents.research_agent import ResearchSession

console = Console()

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The main answer"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"], "description": "Confidence level"},
        "sources": {"type": "array", "items": {"type": "string"}, "description": "Sources used"},
    },
    "required": ["answer", "confidence"],
}


def run_research_agent(assistant: AIAssistant, query: str) -> None:
    session = ResearchSession(query, vector_store=assistant.vector_store)
    result = session.run()
    while result.get("status") == "clarification_pending" and result.get("pending_question"):
        console.print(Panel(
            f"[bold yellow]Clarification needed:[/bold yellow] {result['pending_question']}",
            border_style="yellow",
        ))
        reply = Prompt.ask("[bold green]You[/bold green]").strip()
        if not reply:
            break
        result = session.continue_with_reply(reply)

    console.print(Panel(
        Markdown(result.get("answer", "")),
        title=f"Research Response ({result.get('confidence', '?')} confidence)",
        border_style="magenta",
    ))
    console.print(
        f"[dim]agentic trace: steps={result['steps']} tool_calls="
        f"{[t['action'] for t in result['tool_calls']]} "
        f"tokens={result['tokens']['total_tokens']} "
        f"status={result['status']} degraded={result['degraded']}[/dim]"
    )


def main():
    console.print(console)
    console.print(Panel(
        "[bold cyan]AI Assistant[/bold cyan] powered by Gemini 2.0 Flash\n"
        "Commands: [bold]/ingest <path>[/bold] load documents | "
        "[bold]/research <query>[/bold] agentic research loop | "
        "[bold]/structured[/bold] structured output mode | "
        "[bold]/tools[/bold] toggle tools | "
        "[bold]/quit[/bold] exit",
        title="Welcome",
        border_style="cyan",
    ))

    try:
        assistant = AIAssistant()
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    structured_mode = False
    console.print(f"[green]Vector store loaded with {assistant.vector_store.count} chunks.[/green]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        if user_input.startswith("/research"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]Usage: /research <query>[/red]")
                continue
            with console.status("[cyan]Researching...[/cyan]"):
                run_research_agent(assistant, parts[1])
            continue

        if user_input.startswith("/ingest"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]Usage: /ingest <path_to_file_or_directory>[/red]")
                continue
            path = Path(parts[1])
            if not path.exists():
                console.print(f"[red]Path not found: {path}[/red]")
                continue
            with console.status("[cyan]Ingesting documents...[/cyan]"):
                count = assistant.ingest_documents(path)
            console.print(f"[green]Ingested {count} chunks from {path}[/green]")
            continue

        if user_input == "/structured":
            structured_mode = not structured_mode
            mode_str = "[green]ON[/green]" if structured_mode else "[red]OFF[/red]"
            console.print(f"Structured output mode: {mode_str}")
            continue

        if user_input == "/tools":
            assistant._tools_enabled = not assistant._tools_enabled
            tools_str = "[green]ON[/green]" if assistant._tools_enabled else "[red]OFF[/red]"
            console.print(f"Tool calling: {tools_str}")
            continue

        try:
            with console.status("[cyan]Thinking...[/cyan]"):
                if structured_mode:
                    result = assistant.chat_structured(user_input, STRUCTURED_SCHEMA)
                    console.print(Panel(
                        Markdown(result["response"]),
                        title="Response (Structured)",
                        border_style="blue",
                    ))
                    if result["structured_data"]:
                        import json
                        console.print(Panel(
                            json.dumps(result["structured_data"], indent=2),
                            title="Parsed JSON",
                            border_style="green",
                        ))
                else:
                    result = assistant.chat(user_input)
                    console.print(Panel(
                        Markdown(result["response"]),
                        title="Response",
                        border_style="blue",
                    ))
                    if result["tool_calls"]:
                        for tc in result["tool_calls"]:
                            console.print(f"  [dim]Tool: {tc['name']}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
