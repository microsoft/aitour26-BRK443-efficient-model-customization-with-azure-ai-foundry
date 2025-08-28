"""
Shared utilities for RAFT CLI commands.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import click
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from dotenv_azd import load_azd_env
from openai import AzureOpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from utils import get_env_state_file

# Initialize Rich console
console = Console()

# Configure logging with Rich
import logging
logger = logging.getLogger("raft_cli")


def execute_command(
    command: str, 
    cwd: Optional[str] = None, 
    env_vars: Optional[dict] = None,
    description: Optional[str] = None
) -> Tuple[int, str, str]:
    """
    Execute a shell command and display real-time output.
    
    Args:
        command: The command to execute
        cwd: Working directory for the command
        env_vars: Additional environment variables
        description: Description for logging
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if description:
        logger.info(f"🔄 {description}")
    else:
        logger.info(f"🔄 Executing: {command}")
    
    # Prepare environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    # Ensure subprocess thinks it has a terminal for colored output
    env['FORCE_COLOR'] = '1'
    env['TERM'] = 'xterm-256color'
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        stdout_lines = []
        
        # Read output line by line in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.rstrip()
                stdout_lines.append(line)
                # Display with visual distinction - indented but preserve colors
                # Use rich's Text object to handle ANSI codes properly
                colored_line = Text.from_ansi(line)
                # Create the complete indented line with styling
                indent_text = Text("    │ ", style="dim")
                # Apply dim style to the colored line while preserving original colors
                colored_line.stylize("dim")
                # Combine indent and content in a single Text object
                full_line = indent_text + colored_line
                console.print(full_line)
        
        # Wait for process to complete
        return_code = process.poll()
        stdout = '\n'.join(stdout_lines)
        
        if return_code == 0:
            logger.info(f"✅ Command completed successfully")
        else:
            logger.error(f"❌ Command failed with return code {return_code}")
        
        return return_code, stdout, ""
        
    except Exception as e:
        logger.error(f"❌ Failed to execute command: {e}")
        return 1, "", str(e)


def setup_environment():
    """
    Set up environment variables by loading from various sources.
    """
    logger.info("🔧 Setting up environment variables")
    
    # Load AZD environment
    load_azd_env(quiet=True)
    
    # Load user provided values
    load_dotenv(".env")
    
    # Load variables passed by previous scripts/notebooks
    load_dotenv(get_env_state_file())
    
    return True


def create_azure_openai_client() -> AzureOpenAI:
    """
    Create and configure Azure OpenAI client with proper authentication.
    
    Returns:
        Configured AzureOpenAI client instance
        
    Raises:
        click.ClickException: If required environment variables are missing
    """
    aoai_endpoint = os.getenv("FINETUNE_AZURE_OPENAI_ENDPOINT")
    if not aoai_endpoint:
        raise click.ClickException("❌ FINETUNE_AZURE_OPENAI_ENDPOINT not found in environment")
    
    logger.info(f"🌐 Using Azure OpenAI endpoint: {aoai_endpoint}")
    logger.info("🔐 Authenticating with Azure")
    
    azure_credential = DefaultAzureCredential()
    
    return AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_version="2024-05-01-preview",  # Required for fine-tuning features
        azure_ad_token_provider=get_bearer_token_provider(
            azure_credential, "https://cognitiveservices.azure.com/.default"
        )
    )
