"""
RAFT Infrastructure Check Command

Command for verifying Azure AI service connectivity and endpoint configuration.
"""

import logging
import os
import sys

import rich_click as click
from rich.console import Console

from lib.shared import execute_command, setup_environment, console, logger


def setup_check_environment():
    """
    Set up environment variables for checking infrastructure and return deployment names.
    
    Returns:
        Tuple of (embedding_deployment_name, teacher_deployment_name)
    """
    setup_environment()  # Load all env vars
    
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    teacher_deployment = os.getenv("TEACHER_DEPLOYMENT_NAME")
    
    if not embedding_deployment:
        logger.error("❌ EMBEDDING_DEPLOYMENT_NAME not found in environment")
        raise click.ClickException("Missing EMBEDDING_DEPLOYMENT_NAME environment variable")
    
    if not teacher_deployment:
        logger.error("❌ TEACHER_DEPLOYMENT_NAME not found in environment")
        raise click.ClickException("Missing TEACHER_DEPLOYMENT_NAME environment variable")
    
    logger.info(f"📊 Using embedding model: {embedding_deployment}")
    logger.info(f"🧠 Using teacher model: {teacher_deployment}")
    
    return embedding_deployment, teacher_deployment


def run_infrastructure_tests():
    """Run infrastructure tests to verify endpoints."""
    logger.info("🧪 Running infrastructure tests")
    return_code, _, _ = execute_command(
        "python -m pytest --rootdir=infra/tests/",
        description="Verifying infrastructure endpoints"
    )
    
    if return_code != 0:
        logger.error("❌ Infrastructure tests failed")
        raise click.ClickException("Infrastructure tests failed. Please check your endpoints.")


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
def check(verbose: bool):
    """
    Check infrastructure endpoints and verify Azure AI service connectivity.
    
    This command verifies that all required Azure AI services are properly
    configured and accessible before running dataset generation or other
    operations that depend on these services.
    
    [bold yellow]What it checks:[/bold yellow]
    [dim]•[/dim] Azure OpenAI API endpoints
    [dim]•[/dim] Embedding model deployment
    [dim]•[/dim] Teacher model deployment
    [dim]•[/dim] Authentication and permissions
    
    [bold green]Example:[/bold green]
    [cyan]raft check[/cyan]
    """
    # Configure logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 Verbose logging enabled")
    
    console.print("\n🔍 [bold blue]Infrastructure Checker[/bold blue]\n")
    
    try:
        # Setup environment
        embedding_deployment, teacher_deployment = setup_check_environment()
        
        # Run infrastructure tests
        run_infrastructure_tests()
        
        console.print("\n✅ [bold green]All infrastructure checks passed![/bold green]\n")
        console.print("🎯 [bold]Your Azure AI services are ready for dataset generation.[/bold]")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)
