import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Import rich elements for premium terminal aesthetics
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown

# Load dotenv to find .env file
load_dotenv()

# We need to add the current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import build_workflow, TAVILY_API_KEY

console = Console()

def print_header():
    """Print a beautiful header for the tool."""
    console.print()
    header_content = (
        "[bold cyan]▲ Antigravity Company Project Analyzer ▲[/bold cyan]\n"
        "[dim]LangGraph + Ollama (Mistral) + Tavily Intelligence Agent[/dim]"
    )
    console.print(Panel(header_content, border_style="cyan", expand=False))
    console.print()

def check_requirements() -> bool:
    """Check requirements and display appropriate warnings/status."""
    if not TAVILY_API_KEY or TAVILY_API_KEY.strip() == "" or TAVILY_API_KEY.startswith("your_"):
        warning_msg = (
            "[bold yellow]Notice: Tavily API Key Not Configured[/bold yellow]\n\n"
            "The [cyan]TAVILY_API_KEY[/cyan] is not set in your [bold].env[/bold] file.\n"
            "The analyzer will run in [bold green]Mock Sandbox Mode[/bold green], using local mock data "
            "and Ollama to emulate web search results.\n\n"
            "To use live web data, please get a free key from [link=https://tavily.com]https://tavily.com[/link] "
            "and add it to your [bold].env[/bold] file:\n"
            "TAVILY_API_KEY=your_tavily_api_key"
        )
        console.print(Panel(warning_msg, border_style="yellow", expand=False))
        console.print()
    return True


def save_report(company_name: str, analysis: dict) -> str:
    """Save the generated analysis JSON to a report file."""
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/{company_name.lower().replace(' ', '_')}_analysis.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    return filename

def display_analysis_report(analysis_dict: dict, file_path: str):
    """Print the final analysis report using Rich formatting."""
    company = analysis_dict.get("company_name", "Unknown Company")
    summary = analysis_dict.get("summary", "")
    projects = analysis_dict.get("projects", [])
    services = analysis_dict.get("recommended_services", [])

    console.print(f"\n[bold green]✔ Analysis Completed for {company}![/bold green]\n")
    
    # 1. Summary Panel
    console.print(Panel(
        Markdown(summary),
        title=f"[bold gold1]Overview of {company}[/bold gold1]",
        border_style="gold1"
    ))
    console.print()

    # 2. Projects Table
    table = Table(title="Identified Projects", show_header=True, header_style="bold magenta")
    table.add_column("Project", style="cyan", width=30)
    table.add_column("Description", style="white")
    table.add_column("Citations (Sources)", style="green", width=40)

    for proj in projects:
        proj_name = proj.get("name", "Unnamed Project")
        proj_desc = proj.get("description", "")
        proj_citations = proj.get("citations", [])
        
        citations_str = "\n".join([f"• {url}" for url in proj_citations])
        
        table.add_row(
            proj_name,
            proj_desc.strip(),
            citations_str or "No Citations"
        )
        table.add_section()

    console.print(table)
    console.print()

    # 3. Service Proposals
    if services:
        service_table = Table(title="Recommended Services to Offer", show_header=True, header_style="bold yellow")
        service_table.add_column("Service Proposal", style="yellow", width=30)
        service_table.add_column("Rationale / Why it fits", style="white")
        service_table.add_column("Target Projects", style="cyan", width=30)
        
        for s in services:
            s_name = s.get("service_name", "Unnamed Service")
            s_rat = s.get("rationale", "")
            s_targets = ", ".join(s.get("target_projects", []))
            
            service_table.add_row(s_name, s_rat, s_targets)
            
        console.print(service_table)
        console.print()

    # 4. Save Status
    console.print(Panel(
        f"Full JSON report saved successfully to:\n[bold green]{os.path.abspath(file_path)}[/bold green]",
        title="Saved Report",
        border_style="green",
        expand=False
    ))
    console.print()

def main():
    print_header()
    
    if len(sys.argv) < 2:
        # Prompt user if not passed as CLI argument
        company_name = console.input("[bold yellow]Enter the company name to analyze: [/bold yellow]").strip()
    else:
        company_name = " ".join(sys.argv[1:]).strip()
        
    if not company_name:
        console.print("[red]Error: Company name cannot be empty.[/red]")
        sys.exit(1)
        
    if not check_requirements():
        sys.exit(1)
        
    initial_state = {
        "company_name": company_name,
        "queries": [],
        "search_results": [],
        "graded_results": [],
        "relevance_grades": [],
        "analysis": None,
        "hallucination_review": None,
        "attempts": 0,
        "review_attempts": 0,
        "logs": [],
        "error": None
    }
    
    app = build_workflow()
    
    console.print(f"[bold green]▶[/bold green] Starting project research pipeline for: [bold cyan]{company_name}[/bold cyan]...")
    console.print("[dim]This may take a few moments. Running LangGraph nodes...[/dim]\n")
    
    # Run the graph
    with console.status(f"[bold green]Analyzing {company_name}...[/bold green]", spinner="dots"):
        final_state = app.invoke(initial_state)
        
    if final_state.get("error"):
        console.print(Panel(
            f"[bold red]Workflow Error:[/bold red] {final_state['error']}",
            border_style="red"
        ))
        sys.exit(1)
        
    analysis = final_state.get("analysis")
    
    if not analysis:
        console.print("[red]Error: Graph run finished but no analysis was generated.[/red]")
        sys.exit(1)
        
    # Convert analysis to dictionary
    # Pydantic v2 .model_dump()
    try:
        analysis_dict = analysis.model_dump()
    except AttributeError:
        # Fallback for Pydantic v1
        analysis_dict = analysis.dict()
        
    saved_file = save_report(company_name, analysis_dict)
    display_analysis_report(analysis_dict, saved_file)

if __name__ == "__main__":
    main()
