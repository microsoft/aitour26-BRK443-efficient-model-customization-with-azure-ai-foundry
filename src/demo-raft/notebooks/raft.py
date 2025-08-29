#!/usr/bin/env python3
"""
RAFT CLI Tool

A comprehensive command-line tool for RAFT (Retrieval Augmented Fine Tuning)
operations including dataset generation, fine-tuning, and evaluation.
"""

import logging

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler

from lib.commands.gen import gen
from lib.commands.check import check
from lib.commands.finetune import finetune
from lib.commands.deploy import deploy
from lib.commands.configure import configure
from lib.commands.eval import eval

# Configure Rich Click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = "Try running the '--help' flag for more information."
click.rich_click.ERRORS_EPILOGUE = "To find out more, visit https://github.com/microsoft/aitour26-BRK443"
click.rich_click.COMMAND_GROUPS = {
    "raft.py": [
        {
            "name": "Workflow Commands",
            "commands": ["configure", "gen", "finetune", "deploy", "eval"],
        },
        {
            "name": "Utility Commands", 
            "commands": ["check", "status", "clean"],
        },
    ]
}

# Initialize Rich console
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies").setLevel(logging.WARNING)

@click.group(chain=True)
@click.version_option(version="1.0.0", prog_name="RAFT CLI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """
    [bold blue]RAFT CLI[/bold blue] - [italic]Retrieval Augmented Fine Tuning toolkit[/italic]
    
    A comprehensive tool for generating synthetic datasets, fine-tuning models,
    and evaluating performance using the RAFT methodology with Azure AI services.
    
    [bold yellow]Common Workflow:[/bold yellow]
    [dim]1.[/dim] [cyan]raft configure[/cyan] - Configure AI models and deployments
    [dim]2.[/dim] [cyan]raft check[/cyan] - Verify Azure AI endpoints are working
    [dim]3.[/dim] [cyan]raft gen[/cyan] - Generate synthetic training datasets
    [dim]4.[/dim] [cyan]raft finetune[/cyan] - Fine-tune models with generated data
    [dim]5.[/dim] [cyan]raft deploy[/cyan] - Deploy fine-tuned models to Azure OpenAI
    [dim]6.[/dim] [cyan]raft eval[/cyan] - Evaluate model performance and compare results
    [dim]7.[/dim] [cyan]raft status[/cyan] - Monitor progress and results
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("🔧 [dim]Verbose logging enabled[/dim]")
    
    console.print("🤖 [bold blue]RAFT CLI Toolkit[/bold blue] - Retrieval Augmented Fine Tuning")


# Add the commands
cli.add_command(configure)
cli.add_command(check)
cli.add_command(gen)
cli.add_command(finetune)
cli.add_command(deploy)
cli.add_command(eval)


@click.command()
def status():
    """
    Show current RAFT project status and configuration.
    
    [dim]Displays information about:[/dim]
    • Environment variables and Azure deployments
    • Existing datasets and their status
    • Recent operations and job statuses
    """
    console.print("📊 [bold]RAFT Project Status[/bold]")
    
    # TODO: Implement status checking logic
    # - Check environment variables
    # - Check Azure deployments
    # - Check existing datasets
    # - Show recent operations
    
    console.print("⚠️  Status command not yet implemented")


@click.command()
def clean():
    """
    Clean up generated datasets and temporary files.
    
    [dim]This will remove:[/dim]
    • Generated dataset directories
    • Temporary processing files
    • State files and cached data
    
    [bold red]Warning:[/bold red] This action cannot be undone!
    """
    console.print("🧹 [bold]Cleaning RAFT workspace[/bold]")
    
    # TODO: Implement cleanup logic
    # - Remove dataset directories
    # - Clean temporary files
    # - Reset state files
    
    console.print("⚠️  Clean command not yet implemented")


# Add additional commands
cli.add_command(status)
cli.add_command(clean)


if __name__ == "__main__":
    cli()
