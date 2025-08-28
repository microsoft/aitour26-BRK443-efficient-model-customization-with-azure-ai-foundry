#!/usr/bin/env python3
"""
RAFT CLI Tool

A comprehensive command-line tool for RAFT (Retrieval Augmented Fine Tuning)
operations including dataset generation, fine-tuning, and evaluation.
"""

import logging

import click
from rich.console import Console
from rich.logging import RichHandler

from lib.commands.gen import gen
from lib.commands.finetune import finetune

# Initialize Rich console
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)


@click.group()
@click.version_option(version="1.0.0", prog_name="RAFT CLI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """
    RAFT CLI - Retrieval Augmented Fine Tuning toolkit.
    
    A comprehensive tool for generating synthetic datasets, fine-tuning models,
    and evaluating performance using the RAFT methodology with Azure AI services.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("🔧 [dim]Verbose logging enabled[/dim]")
    
    console.print("🤖 [bold blue]RAFT CLI Toolkit[/bold blue] - Retrieval Augmented Fine Tuning")


# Add the gen and finetune commands
cli.add_command(gen)
cli.add_command(finetune)


@click.command()
def status():
    """Show current RAFT project status and configuration."""
    console.print("📊 [bold]RAFT Project Status[/bold]")
    
    # TODO: Implement status checking logic
    # - Check environment variables
    # - Check Azure deployments
    # - Check existing datasets
    # - Show recent operations
    
    console.print("⚠️  Status command not yet implemented")


@click.command()
def clean():
    """Clean up generated datasets and temporary files."""
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
