"""
RAFT Fine-tuning Command

Command for fine-tuning models using generated datasets.
"""

import json
import logging
import os
import tiktoken
from typing import Optional

import rich_click as click
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from rich.table import Table

from lib.shared import setup_environment, console, logger
from utils import update_state


def calculate_training_cost(training_file_path: str, model_name: str, num_epochs: int = 3) -> tuple:
    """
    Calculate the estimated cost of fine-tuning based on token count.
    
    Args:
        training_file_path: Path to the training data file
        model_name: Name of the model to be fine-tuned
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (num_tokens, total_cost)
    """
    logger.info("💰 Calculating training costs")
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Model {model_name} not found in tiktoken encodings, using o200k_base")
        encoding = tiktoken.get_encoding("o200k_base")

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "role":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    with open(training_file_path, 'r', encoding='utf-8') as f:
        num_tokens = 0
        dataset = [json.loads(line) for line in f]
        messages = [d.get('messages') for d in dataset]
        for message in messages:
            num_tokens += num_tokens_from_messages(message)
    
    # GPT-4o mini training cost per 1K tokens
    training_cost_per_token = 0.003300 / 1000
    total_cost = num_tokens * training_cost_per_token * num_epochs
    
    logger.info(f"📊 Training data contains {num_tokens:,} tokens")
    logger.info(f"💵 Estimated cost for {num_epochs} epochs: ${total_cost:.2f} USD")
    
    return num_tokens, total_cost


def upload_training_files(client: AzureOpenAI, training_file_path: str, validation_file_path: str) -> tuple:
    """
    Upload training and validation files to Azure OpenAI.
    
    Args:
        client: Azure OpenAI client
        training_file_path: Path to training data
        validation_file_path: Path to validation data
        
    Returns:
        Tuple of (training_file_id, validation_file_id)
    """
    logger.info("📤 Uploading training files to Azure OpenAI")
    
    # Upload training file
    logger.info(f"📁 Uploading training file: {training_file_path}")
    with open(training_file_path, "rb") as f:
        training_response = client.files.create(file=f, purpose="fine-tune")
    training_file_id = training_response.id
    logger.info(f"✅ Training file uploaded with ID: {training_file_id}")
    
    # Upload validation file
    logger.info(f"📁 Uploading validation file: {validation_file_path}")
    with open(validation_file_path, "rb") as f:
        validation_response = client.files.create(file=f, purpose="fine-tune")
    validation_file_id = validation_response.id
    logger.info(f"✅ Validation file uploaded with ID: {validation_file_id}")
    
    return training_file_id, validation_file_id


def create_finetuning_job(
    client: AzureOpenAI,
    training_file_id: str,
    validation_file_id: str,
    model_name: str,
    seed: int = 105
) -> str:
    """
    Create a fine-tuning job.
    
    Args:
        client: Azure OpenAI client
        training_file_id: ID of the uploaded training file
        validation_file_id: ID of the uploaded validation file
        model_name: Name of the base model to fine-tune
        seed: Random seed for reproducibility
        
    Returns:
        Job ID of the created fine-tuning job
    """
    logger.info("🚀 Creating fine-tuning job")
    
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model_name,
        seed=seed
    )
    
    job_id = response.id
    
    logger.info(f"✅ Fine-tuning job created successfully")
    logger.info(f"📋 Job ID: {job_id}")
    logger.info(f"📊 Status: {response.status}")
    logger.info(f"🤖 Base model: {response.model}")
    
    return job_id


def display_job_summary(job_id: str, model_name: str, num_tokens: int, total_cost: float):
    """Display a summary of the fine-tuning job."""
    table = Table(title="Fine-tuning Job Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Job ID", job_id)
    table.add_row("Base Model", model_name)
    table.add_row("Training Tokens", f"{num_tokens:,}")
    table.add_row("Estimated Cost", f"${total_cost:.2f} USD")
    table.add_row("Status", "submitted")
    
    console.print(table)


@click.command()
@click.option("--model", help="Base model name to fine-tune (defaults from state)")
@click.option("--epochs", default=3, help="Number of training epochs to run")
@click.option("--seed", default=105, help="Random seed for reproducible results")
@click.option("--dry-run", is_flag=True, help="Calculate costs without starting the job")
@click.option("--training-file", help="Override default training file path")
@click.option("--validation-file", help="Override default validation file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable detailed logging output")
def finetune(
    model: Optional[str],
    epochs: int,
    seed: int,
    dry_run: bool,
    training_file: Optional[str],
    validation_file: Optional[str],
    verbose: bool
):
    """
    Fine-tune a model using [bold green]generated training data[/bold green].
    
    This command uploads training datasets to Azure OpenAI and creates a 
    fine-tuning job for knowledge distillation. Uses datasets from the 
    [cyan]gen[/cyan] command by default.
    
    [bold yellow]Process Overview:[/bold yellow]
    [dim]1.[/dim] Load training and validation datasets
    [dim]2.[/dim] Calculate token usage and estimated costs
    [dim]3.[/dim] Upload files to Azure OpenAI service
    [dim]4.[/dim] Create and monitor fine-tuning job
    
    [bold green]Example:[/bold green]
    [cyan]raft finetune --epochs 5 --dry-run[/cyan]
    """
    # Configure logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 Verbose logging enabled")
    
    console.print("\n🎯 [bold blue]RAFT Model Fine-tuning[/bold blue]\n")
    
    try:
        # Setup environment
        setup_environment()
        
        # Get model name from state or parameter
        student_model_name = model or os.getenv("STUDENT_MODEL_NAME")
        if not student_model_name:
            logger.error("❌ No student model specified. Use --model or set STUDENT_MODEL_NAME in environment")
            raise click.ClickException("Missing student model name")
        
        logger.info(f"🤖 Fine-tuning model: {student_model_name}")
        
        # Get dataset files from state or parameters
        if training_file and validation_file:
            training_file_path = training_file
            validation_file_path = validation_file
            logger.info("📁 Using provided training and validation files")
        else:
            # Use files from state (generated by gen command)
            training_file_path = os.getenv("DATASET_TRAIN_PATH")
            validation_file_path = os.getenv("DATASET_VALID_PATH")
            
            if not training_file_path or not validation_file_path:
                logger.error("❌ No training/validation files found in state")
                logger.error("💡 Run 'raft gen' first or specify files with --training-file and --validation-file")
                raise click.ClickException("Missing training data files")
        
        logger.info(f"📂 Training file: {training_file_path}")
        logger.info(f"📂 Validation file: {validation_file_path}")
        
        # Verify files exist
        if not os.path.exists(training_file_path):
            raise click.ClickException(f"Training file not found: {training_file_path}")
        if not os.path.exists(validation_file_path):
            raise click.ClickException(f"Validation file not found: {validation_file_path}")
        
        # Calculate training costs
        num_tokens, total_cost = calculate_training_cost(training_file_path, student_model_name, epochs)
        
        if dry_run:
            logger.info("🔍 Dry run mode - showing cost estimation only")
            display_job_summary("DRY-RUN", student_model_name, num_tokens, total_cost)
            console.print("\n💡 [bold]To start the actual fine-tuning job, run without --dry-run[/bold]")
            return
        
        # Get Azure OpenAI endpoint
        aoai_endpoint = os.getenv("FINETUNE_AZURE_OPENAI_ENDPOINT")
        if not aoai_endpoint:
            logger.error("❌ FINETUNE_AZURE_OPENAI_ENDPOINT not found in environment")
            raise click.ClickException("Missing Azure OpenAI endpoint configuration")
        
        logger.info(f"🌐 Using Azure OpenAI endpoint: {aoai_endpoint}")
        
        # Create Azure OpenAI client
        logger.info("🔐 Authenticating with Azure")
        azure_credential = DefaultAzureCredential()
        
        client = AzureOpenAI(
            azure_endpoint=aoai_endpoint,
            api_version="2024-05-01-preview",  # Required for fine-tuning features
            azure_ad_token_provider=get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            )
        )
        
        # Upload training files
        training_file_id, validation_file_id = upload_training_files(
            client, training_file_path, validation_file_path
        )
        
        # Create fine-tuning job
        job_id = create_finetuning_job(
            client, training_file_id, validation_file_id, student_model_name, seed
        )
        
        # Update state with job information
        logger.info("💾 Updating state with fine-tuning job information")
        update_state("STUDENT_OPENAI_JOB_ID", job_id)
        update_state("STUDENT_OPENAI_TRAINING_FILE_ID", training_file_id)
        update_state("STUDENT_OPENAI_VALIDATION_FILE_ID", validation_file_id)
        update_state("STUDENT_MODEL_BASE_NAME", student_model_name)
        
        # Display summary
        console.print("\n✅ [bold green]Fine-tuning Job Created Successfully![/bold green]\n")
        display_job_summary(job_id, student_model_name, num_tokens, total_cost)
        
        console.print(f"\n🎯 [bold]Next Steps:[/bold]")
        console.print("   • Monitor the fine-tuning job status in Azure OpenAI Studio")
        console.print("   • The job will take some time to complete (typically 10-30 minutes)")
        console.print("   • Once complete, you can deploy the fine-tuned model")
        console.print(f"   • Job ID: [cyan]{job_id}[/cyan]")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if verbose:
            console.print_exception()
        raise click.ClickException(f"Fine-tuning failed: {e}")
