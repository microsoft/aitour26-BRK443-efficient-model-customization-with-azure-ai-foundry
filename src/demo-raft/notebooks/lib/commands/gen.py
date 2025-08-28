"""
RAFT Dataset Generation Command

Command for generating synthetic datasets using RAFT methodology.
"""

import json
import os
import subprocess
import sys
import logging
from math import ceil
from pathlib import Path
from typing import Optional, Tuple

import rich_click as click
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from dotenv_azd import load_azd_env
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from lib.shared import execute_command, setup_environment, console, logger
from utils import get_env_state_file, update_state


def setup_gen_environment() -> Tuple[str, str]:
    """
    Set up environment variables specific to generation and return deployment names.
    
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


def setup_raft_repository():
    """Set up the RAFT repository."""
    logger.info("📦 Setting up RAFT repository")
    execute_command(
        "./setup_raft.sh",
        description="Setting up RAFT repository"
    )


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


def create_dataset_directory(ds_path: str):
    """Create dataset directory."""
    logger.info(f"📁 Creating dataset directory: {ds_path}")
    Path(ds_path).mkdir(parents=True, exist_ok=True)


def run_raft_generation(
    doc_path: str,
    ds_path: str,
    raft_questions: int,
    qa_threshold: int,
    embedding_deployment: str,
    teacher_deployment: str
):
    """Run the RAFT dataset generation."""
    logger.info("🤖 Starting RAFT dataset generation")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating synthetic dataset...", total=None)
        
        # Get Azure environment name for env file
        _, azure_env_name, _ = execute_command(
            "azd env get-value AZURE_ENV_NAME || echo ''",
            description="Getting Azure environment name"
        )
        azure_env_name = azure_env_name.strip()
        azure_env_file = f".azure/{azure_env_name}/.env" if azure_env_name else ""
        
        # Prepare environment files
        execute_command("touch .env", description="Ensuring .env file exists")
        
        # Build the RAFT command
        env_files = f".env {get_env_state_file()}"
        if azure_env_file and Path(azure_env_file).exists():
            env_files += f" {azure_env_file}"
        
        raft_command = (
            f"env $(cat {env_files}) python3 .gorilla/raft/raft.py "
            f"--datapath \"{doc_path}\" "
            f"--output {ds_path} "
            f"--distractors 3 "
            f"--doctype pdf "
            f"--chunk_size 512 "
            f"--questions {raft_questions} "
            f"--workers 2 "
            f"--system-prompt-key gpt "
            f"--completion_model {teacher_deployment} "
            f"--embedding_model {embedding_deployment} "
            f"--qa-threshold {qa_threshold} "
            f"--completion-env-prefix TEACHER"
        )
        
        return_code, stdout, stderr = execute_command(
            raft_command,
            description=f"Generating {qa_threshold} Q&A pairs using RAFT"
        )
        
        progress.update(task, completed=True)
        
        if return_code != 0:
            logger.error("❌ RAFT generation failed")
            raise click.ClickException("RAFT dataset generation failed")
        
        logger.info("✅ RAFT dataset generation completed successfully")


def export_to_jsonl(ds_path: str, ds_name: str) -> str:
    """Export Arrow format to JSONL."""
    logger.info("📄 Exporting dataset to JSONL format")
    
    raft_arrow_file = f"{ds_path}/data-00000-of-00001.arrow"
    dataset_path_hf = f"{ds_path}-files/{ds_name}-hf.full.jsonl"
    
    # Create output directory
    Path(f"{ds_path}-files").mkdir(parents=True, exist_ok=True)
    
    return_code, _, _ = execute_command(
        f"python .gorilla/raft/format.py "
        f"--input {raft_arrow_file} "
        f"--output {dataset_path_hf} "
        f"--output-format hf",
        description="Converting Arrow format to JSONL"
    )
    
    if return_code != 0:
        logger.error("❌ Failed to export to JSONL")
        raise click.ClickException("Failed to export dataset to JSONL")
    
    return dataset_path_hf


def display_dataset_sample(dataset_path_hf: str):
    """Display a sample from the dataset."""
    logger.info("🔍 Loading dataset sample")
    
    try:
        hf_full_df = pd.read_json(dataset_path_hf, lines=True)
        logger.info(f"📊 Dataset contains {len(hf_full_df)} records")
        
        # Display sample information
        if len(hf_full_df) > 0:
            sample = hf_full_df.iloc[2] if len(hf_full_df) > 2 else hf_full_df.iloc[0]
            
            table = Table(title="Dataset Sample")
            table.add_column("Field", style="cyan")
            table.add_column("Preview", style="white")
            
            table.add_row("Question", sample.question[:100] + "..." if len(sample.question) > 100 else sample.question)
            table.add_row("Oracle Context", sample.oracle_context[:100] + "..." if len(sample.oracle_context) > 100 else sample.oracle_context)
            table.add_row("CoT Answer", sample.cot_answer[:100] + "..." if len(sample.cot_answer) > 100 else sample.cot_answer)
            
            console.print(table)
        
        return hf_full_df
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset sample: {e}")
        raise click.ClickException("Failed to load dataset sample")


def split_dataset(
    hf_full_df: pd.DataFrame,
    ds_path: str,
    ds_name: str,
    finetuning_train_split: float,
    finetuning_valid_split: float
) -> Tuple[str, str, str]:
    """Split dataset into train/validation/evaluation sets."""
    logger.info("✂️  Splitting dataset into train/validation/evaluation sets")
    
    # Define file paths
    dataset_path_hf_train = f"{ds_path}-files/{ds_name}-hf.train.jsonl"
    dataset_path_hf_valid = f"{ds_path}-files/{ds_name}-hf.valid.jsonl"
    dataset_path_hf_eval = f"{ds_path}-files/{ds_name}-hf.eval.jsonl"
    
    # Calculate splits
    samples_count = len(hf_full_df)
    splits = [
        int(finetuning_train_split * samples_count),
        int((finetuning_train_split + finetuning_valid_split) * samples_count)
    ]
    
    logger.info(f"📊 Splitting {samples_count} samples at positions: {splits}")
    
    # Split the dataframe
    hf_train_df, hf_valid_df, hf_eval_df = np.split(hf_full_df, splits)
    
    # Save splits
    hf_train_df.to_json(dataset_path_hf_train, orient="records", lines=True)
    hf_valid_df.to_json(dataset_path_hf_valid, orient="records", lines=True)
    hf_eval_df.to_json(dataset_path_hf_eval, orient="records", lines=True)
    
    logger.info(f"✅ Train set: {len(hf_train_df)} samples")
    logger.info(f"✅ Validation set: {len(hf_valid_df)} samples")
    logger.info(f"✅ Evaluation set: {len(hf_eval_df)} samples")
    
    return dataset_path_hf_train, dataset_path_hf_valid, dataset_path_hf_eval


def export_finetuning_format(
    dataset_path_hf_train: str,
    dataset_path_hf_valid: str,
    ds_path: str,
    ds_name: str,
    format_type: str
):
    """Export training and validation splits to fine-tuning format."""
    logger.info(f"🔄 Exporting to {format_type} fine-tuning format")
    
    dataset_path_ft_train = f"{ds_path}-files/{ds_name}-ft.train.jsonl"
    dataset_path_ft_valid = f"{ds_path}-files/{ds_name}-ft.valid.jsonl"
    
    # Export training set
    execute_command(
        f"python .gorilla/raft/format.py "
        f"--input {dataset_path_hf_train} "
        f"--input-type jsonl "
        f"--output {dataset_path_ft_train} "
        f"--output-format {format_type} "
        f"--output-completion-prompt-column text "
        f"--output-completion-completion-column ground_truth",
        description="Exporting training set to fine-tuning format"
    )
    
    # Export validation set
    execute_command(
        f"python .gorilla/raft/format.py "
        f"--input {dataset_path_hf_valid} "
        f"--input-type jsonl "
        f"--output {dataset_path_ft_valid} "
        f"--output-format {format_type} "
        f"--output-completion-prompt-column text "
        f"--output-completion-completion-column ground_truth",
        description="Exporting validation set to fine-tuning format"
    )
    
    logger.info("✅ Fine-tuning format export completed")
    return dataset_path_ft_train, dataset_path_ft_valid


def reformat_datasets(dataset_path_ft_train: str, dataset_path_ft_valid: str, ds_path: str, ds_name: str):
    """Reformat datasets to more fine-tuning friendly format."""
    logger.info("🔄 Reformatting datasets for fine-tuning")
    
    try:
        from lib.reformat_jsonl import reformat_jsonl_file
        
        dataset_path_ft_train_v2 = f"{ds_path}-files/{ds_name}-ft.train.v2.jsonl"
        dataset_path_ft_valid_v2 = f"{ds_path}-files/{ds_name}-ft.valid.v2.jsonl"
        
        reformat_jsonl_file(dataset_path_ft_train, dataset_path_ft_train_v2)
        reformat_jsonl_file(dataset_path_ft_valid, dataset_path_ft_valid_v2)
        
        logger.info("✅ Dataset reformatting completed")
        return dataset_path_ft_train_v2, dataset_path_ft_valid_v2
        
    except ImportError:
        logger.warning("⚠️  Reformatting library not available, skipping reformatting step")
        return dataset_path_ft_train, dataset_path_ft_valid


@click.command()
@click.option("--ds-name", default="zava-articles", help="Name for the generated dataset")
@click.option("--doc-path", default="sample_data/zava-articles", help="Path to source documents")
@click.option("--format", "format_type", default="chat", help="Output format for fine-tuning")
@click.option("--train-split", default=0.8, help="Training data split ratio")
@click.option("--valid-split", default=0.1, help="Validation data split ratio") 
@click.option("--finetuning-threshold", default=369, help="Minimum samples required for fine-tuning")
@click.option("--raft-questions", default=2, help="Number of questions generated per document chunk")
@click.option("--citation-format", 
              type=click.Choice(['legacy-xml-tag', 'md-dash-list'], case_sensitive=False),
              default="legacy-xml-tag", 
              help="Citation format: [green]legacy-xml-tag[/green] (no reformatting) or [blue]md-dash-list[/blue] (reformat)")
@click.option("--skip-setup", is_flag=True, help="Skip RAFT repository setup")
@click.option("--skip-tests", is_flag=True, help="Skip infrastructure endpoint tests")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
def gen(
    ds_name: str,
    doc_path: str,
    format_type: str,
    train_split: float,
    valid_split: float,
    finetuning_threshold: int,
    raft_questions: int,
    citation_format: str,
    skip_setup: bool,
    skip_tests: bool,
    verbose: bool
):
    """
    Generate synthetic datasets using [bold blue]RAFT[/bold blue] methodology.
    
    This command analyzes source documents and generates question-answer pairs
    suitable for fine-tuning smaller language models. Uses Azure AI models
    to create high-quality synthetic training data.
    
    [bold yellow]Process Overview:[/bold yellow]
    [dim]1.[/dim] Setup RAFT repository and verify Azure endpoints
    [dim]2.[/dim] Generate Q&A pairs from document chunks  
    [dim]3.[/dim] Split data into training/validation/evaluation sets
    [dim]4.[/dim] Export in fine-tuning compatible formats
    
    [bold green]Example:[/bold green]
    [cyan]raft gen --ds-name my-dataset --raft-questions 3[/cyan]
    """
    # Configure logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 Verbose logging enabled")
    
    console.print("\n🚀 [bold blue]RAFT Dataset Generator[/bold blue]\n")
    
    try:
        # Setup environment
        embedding_deployment, teacher_deployment = setup_gen_environment()
        
        # Setup RAFT repository
        if not skip_setup:
            setup_raft_repository()
        else:
            logger.info("⏭️  Skipping RAFT repository setup")
        
        # Run infrastructure tests
        if not skip_tests:
            run_infrastructure_tests()
        else:
            logger.info("⏭️  Skipping infrastructure tests")
        
        # Calculate QA threshold
        qa_threshold = ceil(finetuning_threshold / train_split)
        logger.info(f"📊 Calculated QA threshold: {qa_threshold}")
        
        # Setup dataset paths
        ds_path = f"dataset/{ds_name}"
        ds_output_file = f"{ds_path}.jsonl"
        
        # Update state
        update_state("DATASET_NAME", ds_name)
        logger.info(f"📝 Creating dataset: {ds_name}")
        
        # Create dataset directory
        create_dataset_directory(ds_path)
        
        # Run RAFT generation
        run_raft_generation(
            doc_path, ds_path, raft_questions, qa_threshold,
            embedding_deployment, teacher_deployment
        )
        
        # Export to JSONL
        dataset_path_hf = export_to_jsonl(ds_path, ds_name)
        
        # Display dataset sample
        hf_full_df = display_dataset_sample(dataset_path_hf)
        
        # Split dataset
        dataset_path_hf_train, dataset_path_hf_valid, dataset_path_hf_eval = split_dataset(
            hf_full_df, ds_path, ds_name, train_split, valid_split
        )
        
        # Export to fine-tuning format
        dataset_path_ft_train, dataset_path_ft_valid = export_finetuning_format(
            dataset_path_hf_train, dataset_path_hf_valid, ds_path, ds_name, format_type
        )
        
        # Conditionally reformat datasets based on citation format
        if citation_format == "md-dash-list":
            logger.info("🔄 Citation format set to md-dash-list, reformatting datasets")
            dataset_path_ft_train_v2, dataset_path_ft_valid_v2 = reformat_datasets(
                dataset_path_ft_train, dataset_path_ft_valid, ds_path, ds_name
            )
            final_train_path = dataset_path_ft_train_v2
            final_valid_path = dataset_path_ft_valid_v2
            train_label = "Training Set (v2)"
            valid_label = "Validation Set (v2)"
        else:
            logger.info("🔄 Citation format set to legacy-xml-tag, skipping reformatting")
            final_train_path = dataset_path_ft_train
            final_valid_path = dataset_path_ft_valid
            train_label = "Training Set"
            valid_label = "Validation Set"
        
        # Update state with generated file paths
        logger.info("💾 Updating state with generated file paths")
        update_state("DATASET_TRAIN_PATH", final_train_path)
        update_state("DATASET_VALID_PATH", final_valid_path)
        update_state("DATASET_EVAL_PATH", dataset_path_hf_eval)
        update_state("DATASET_PATH_HF_TRAIN", dataset_path_hf_train)
        update_state("DATASET_PATH_HF_VALID", dataset_path_hf_valid)
        update_state("DATASET_PATH_HF_EVAL", dataset_path_hf_eval)
        update_state("CITATION_FORMAT", citation_format)
        
        # Success summary
        console.print("\n✅ [bold green]Dataset Generation Complete![/bold green]\n")
        
        summary_table = Table(title="Generated Files")
        summary_table.add_column("File Type", style="cyan")
        summary_table.add_column("Path", style="white")
        
        summary_table.add_row(train_label, final_train_path)
        summary_table.add_row(valid_label, final_valid_path)
        summary_table.add_row("Evaluation Set", dataset_path_hf_eval)
        
        console.print(summary_table)
        
        console.print(f"\n🎯 [bold]Next Steps:[/bold]")
        console.print("   • Use the generated training/validation sets for fine-tuning")
        console.print("   • Run evaluation on the evaluation set")
        console.print("   • Consider running [cyan]2_finetune_oai.ipynb[/cyan] for OpenAI fine-tuning")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)
