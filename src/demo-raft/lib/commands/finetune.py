"""
RAFT Fine-tuning Command

Command for fine-tuning models using generated datasets.
"""

import json
import logging
import os
import time
import hashlib
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from litellm import FileTypes
from rich.table import Table

from typing import Optional

import tiktoken
import rich_click as click
from openai import AzureOpenAI

from lib.shared import setup_environment, console, logger, create_azure_openai_client
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


def sdk_wait_for_processing(client: AzureOpenAI, file_id: str, timeout: int = 300) -> None:
    """
    Call the SDK's Files.wait_for_processing helper once and raise a ClickException on error.
    """
    logger.info(f"⏳ Waiting (via SDK helper) for file {file_id} to be processed (timeout {timeout}s)")
    try:
        client.files.wait_for_processing(file_id, timeout=timeout)
        logger.info(f"✅ File {file_id} processed (via SDK helper)")
    except Exception as e:
        logger.debug(f"SDK wait_for_processing failed for {file_id}: {e}")
        raise click.ClickException(f"File import failed or wait helper errored for {file_id}: {e}")


def find_existing_file(client: AzureOpenAI, local_path: str, expected_filename: Optional[str] = None):
    """
    Look for an already uploaded file in Azure OpenAI that matches either the
    expected_filename (preferred) or the local file basename and size.
    Returns (file_id, status) or (None, None).
    """
    basename = os.path.basename(local_path)
    try:
        local_size = os.path.getsize(local_path)
    except OSError:
        local_size = None

    logger.info(f"🔎 Checking for existing Azure file: {basename}")
    try:
        files_iter = client.files.list()
    except Exception as e:
        logger.debug(f"Failed to list Azure files: {e}")
        return None, None

    seen_ids = set()
    for f in files_iter:
        name = f.filename or f.name
        size = f.bytes or f.size
        purpose = f.purpose
        status = f.status
        fid = f.id

        # protect against paginators that may repeat the last page
        if fid in seen_ids:
            logger.debug(f"Encountered duplicate file id {fid}; stopping iteration")
            break
        seen_ids.add(fid)

        logger.debug(f"Found Azure file: {name} (id={fid}, status={status})")

        if not name or purpose != "fine-tune":
            continue

        # If expected_filename provided, prefer exact match on that
        try:
            if expected_filename and name == expected_filename:
                logger.info(f"⚠️  Found existing Azure file by expected name: {name} (id={fid}, status={status})")
                return fid, status
            # Otherwise fall back to basename + size match
            if name == basename and (local_size is None or size == local_size):
                logger.info(f"⚠️  Found existing Azure file: {name} (id={fid}, status={status})")
                return fid, status
        except Exception:
            continue

    return None, None


def upload_finetuning_file(client: AzureOpenAI, local_path: str, file_type: str = "training") -> str:
    """
    Upload or reuse a single fine-tuning file. Returns the Azure file id.
    file_type should be 'training' or 'validation' and is used in logs.
    """
    logger.info(f"🔁 Processing {file_type} file: {local_path}")

    # compute short sha1 of file content to include in uploaded filename
    sha = hashlib.sha1()
    with open(local_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    short_hash = sha.hexdigest()[:7]
    # include dataset name if present
    ds_label = os.getenv("DATASET_NAME") or os.path.basename(os.path.dirname(local_path)) or "dataset"
    ds_label = ds_label.replace(" ", "-")
    upload_name = f"raft-{ds_label}-{file_type}-{short_hash}"

    # Check for an existing file matching the upload_name first
    file_id, status = find_existing_file(client, local_path, expected_filename=upload_name)
    if file_id:
        logger.info(f"📁 Reusing existing {file_type} file: {file_id}")
        if status and str(status).lower() not in ("processed", "uploaded", "succeeded", "available", "ready"):
            sdk_wait_for_processing(client, file_id)
        return file_id

    logger.info(f"📁 Uploading {file_type} file: {local_path} as {upload_name}")
    with open(local_path, "rb") as f:
        # include filename in the upload so it's recorded in Azure
        ft_file: FileTypes = (upload_name, f)
        resp = client.files.create(file=ft_file, purpose="fine-tune")
    file_id = resp.id
    logger.info(f"✅ {file_type.capitalize()} file uploaded with ID: {file_id}")
    sdk_wait_for_processing(client, file_id)
    return file_id


def upload_training_files(client: AzureOpenAI, training_file_path: str, validation_file_path: str) -> tuple:
    """
    Upload training and validation files to Azure OpenAI and wait for imports to complete.
    Uses upload_finetuning_file to avoid duplicating logic.
    """
    logger.info("📤 Uploading training files to Azure OpenAI")

    training_file_id = upload_finetuning_file(client, training_file_path, file_type="training")
    validation_file_id = upload_finetuning_file(client, validation_file_path, file_type="validation")

    return training_file_id, validation_file_id


def find_existing_finetune_job(client: AzureOpenAI, training_file_id: str, validation_file_id: str, model_name: str):
    """
    Search existing fine-tuning jobs for a job that matches the training/validation file IDs and model.
    Returns (job_id, status) or (None, None).
    """
    logger.info("🔎 Searching for existing fine-tuning jobs")
    try:
        jobs_iter = client.fine_tuning.jobs.list()
    except Exception as e:
        logger.debug(f"Failed to list fine-tuning jobs: {e}")
        return None, None

    for job in jobs_iter:

        t = job.training_file
        v = job.validation_file
        m = job.model
        jid = job.id
        status = job.status

        if t == training_file_id and v == validation_file_id and m.startswith(model_name):
            logger.info(f"⚠️  Found existing fine-tuning job: {jid} (status={status})")
            return jid, status

    return None, None


def create_finetuning_job(
    client: AzureOpenAI,
    training_file_id: str,
    validation_file_id: str,
    model_name: str,
    seed: int = 105,
    dataset_name: Optional[str] = None
) -> str:
    """
    Create a fine-tuning job, but reuse an existing matching job if one already exists.
    The job name will include the dataset name when provided.
    """
    logger.info("🚀 Creating fine-tuning job")

    # Check if an equivalent job already exists
    existing_job_id, existing_status = find_existing_finetune_job(client, training_file_id, validation_file_id, model_name)
    if existing_job_id:
        logger.info(f"♻️  Reusing existing fine-tuning job {existing_job_id} (status={existing_status})")
        return existing_job_id

    # Build a friendly job name that includes the dataset name and a short hash
    ds_label = dataset_name or os.getenv("DATASET_NAME") or os.path.splitext(os.path.basename(training_file_id))[0]
    ds_label = ds_label.replace(" ", "-") if ds_label else "dataset"
    job_name_suffix = f"{ds_label}"

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model_name,
        seed=seed,
        job_name_suffix=job_name_suffix
    )
    
    job_id = response.id
    
    logger.info(f"✅ Fine-tuning job created successfully")
    logger.info(f"📋 Job ID: {job_id}")
    logger.info(f"📊 Status: {response.status}")
    logger.info(f"🤖 Base model: {response.model}")
    logger.info(f"🏷️ Job name suffix: {job_name_suffix}")
    
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
        
        # Create Azure OpenAI client
        client = create_azure_openai_client()
        
        # Upload training files
        training_file_id, validation_file_id = upload_training_files(
            client, training_file_path, validation_file_path
        )
        
        # Create fine-tuning job (include dataset name)
        dataset_name = os.getenv("DATASET_NAME") or os.path.splitext(os.path.basename(training_file_path))[0]
        job_id = create_finetuning_job(
            client, training_file_id, validation_file_id, student_model_name, seed, dataset_name
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
