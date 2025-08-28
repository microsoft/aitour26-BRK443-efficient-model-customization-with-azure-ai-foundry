"""
RAFT (Retrieval Augmented Fine Tuning) Wrapper

This module provides a Python wrapper for the UC Berkeley Gorilla RAFT command line tool.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union, Literal
from dotenv import load_dotenv
from dotenv_azd import load_azd_env


def run_raft_generation(
    datapath: str,
    output: str,
    output_format: Literal["hf", "completion", "chat", "eval"] = "hf",
    output_type: Literal["parquet", "jsonl"] = "jsonl",
    output_chat_system_prompt: Optional[str] = None,
    output_completion_prompt_column: str = "prompt",
    output_completion_completion_column: str = "completion",
    distractors: int = 3,
    p: float = 1.0,
    questions: int = 5,
    chunk_size: int = 512,
    doctype: Literal["api", "pdf", "json", "txt"] = "pdf",
    openai_key: Optional[str] = None,
    embedding_model: str = "text-embedding-ada-002",
    completion_model: str = "gpt-4",
    system_prompt_key: Literal["gpt", "llama"] = "gpt",
    workers: int = 2,
    auto_clean_checkpoints: bool = False,
    qa_threshold: Optional[int] = None,
    embedding_env_prefix: str = "EMBEDDING",
    completion_env_prefix: str = "COMPLETION",
    env_files: list = [".env"],
    raft_script_path: str = ".gorilla/raft/raft.py"
) -> subprocess.CompletedProcess:
    """
    Run the RAFT synthetic dataset generation process.
    
    RAFT (Retrieval Augmented Fine Tuning) uses a large LLM to analyze documents
    and generate questions and answers for fine-tuning smaller models.
    
    Args:
        datapath: Path to document(s) - file or folder containing documents
        output: Path where the generated dataset will be saved
        output_format: Format of the output dataset
        output_type: Type to export the dataset to
        output_chat_system_prompt: System prompt for chat format
        output_completion_prompt_column: Prompt column name for completion format
        output_completion_completion_column: Completion column name for completion format
        distractors: Number of distractor documents per data point
        p: Percentage that oracle document is included in context
        questions: Number of data points to generate per chunk
        chunk_size: Size of each chunk in tokens
        doctype: Type of documents being processed
        openai_key: OpenAI API key (if not in environment)
        embedding_model: Model for encoding document chunks
        completion_model: Model for generating questions and answers
        system_prompt_key: System prompt type to use
        workers: Number of worker threads
        auto_clean_checkpoints: Whether to auto-clean checkpoints
        qa_threshold: Number of Q/A samples after which to stop generation
        embedding_env_prefix: Environment variable prefix for embedding model
        completion_env_prefix: Environment variable prefix for completion model
        env_files: List of environment files to load
        raft_script_path: Path to the RAFT script
        
    Returns:
        subprocess.CompletedProcess: Result of the RAFT process execution
        
    Raises:
        FileNotFoundError: If the RAFT script is not found
        subprocess.CalledProcessError: If the RAFT process fails
    """
    
    # Load environment variables from specified files
    for env_file in env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)
            print(f"Loaded environment variables from {env_file}")
    
    # Check if RAFT script exists
    raft_path = Path(raft_script_path)
    if not raft_path.exists():
        raise FileNotFoundError(f"RAFT script not found at {raft_script_path}")
    
    # Build the command
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(raft_path),
        "--datapath", str(datapath),
        "--output", str(output),
        "--output-format", output_format,
        "--output-type", output_type,
        "--output-completion-prompt-column", output_completion_prompt_column,
        "--output-completion-completion-column", output_completion_completion_column,
        "--distractors", str(distractors),
        "--p", str(p),
        "--questions", str(questions),
        "--chunk_size", str(chunk_size),
        "--doctype", doctype,
        "--embedding_model", embedding_model,
        "--completion_model", completion_model,
        "--system-prompt-key", system_prompt_key,
        "--workers", str(workers),
        "--auto-clean-checkpoints", str(auto_clean_checkpoints).lower(),
        "--embedding-env-prefix", embedding_env_prefix,
        "--completion-env-prefix", completion_env_prefix,
    ]
    
    # Add optional parameters
    if output_chat_system_prompt:
        cmd.extend(["--output-chat-system-prompt", output_chat_system_prompt])
    
    if openai_key:
        cmd.extend(["--openai_key", openai_key])
    
    if qa_threshold is not None:
        cmd.extend(["--qa-threshold", str(qa_threshold)])
    
    print(f"Running RAFT generation with command:")
    print(" ".join(cmd))
    
    # Run the command with current environment variables
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy()  # Pass current environment including loaded .env vars
        )
        
        print("RAFT generation completed successfully!")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        return result
        
    except subprocess.CalledProcessError as e:
        print(f"RAFT generation failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        raise


def run_raft_from_notebook_params(
    ds_name: str,
    doc_path: str,
    format: str = "chat",
    finetuning_train_split: float = 0.8,
    finetuning_threshold: int = 369,
    raft_questions: int = 2,
    teacher_deployment_name: Optional[str] = None,
    embedding_deployment_name: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Run RAFT generation using parameters similar to those used in the Jupyter notebook.
    
    This is a convenience function that matches the parameter style used in the 
    1_gen.ipynb notebook.
    
    Args:
        ds_name: Name of the dataset
        doc_path: Path to the documents
        format: Output format for the dataset
        finetuning_train_split: Training split ratio
        finetuning_threshold: Minimum threshold for fine-tuning samples
        raft_questions: Number of questions to generate per chunk
        teacher_deployment_name: Name of the teacher model deployment
        embedding_deployment_name: Name of the embedding model deployment
        
    Returns:
        subprocess.CompletedProcess: Result of the RAFT process execution
    """
    
    # Load environment variables to get deployment names if not provided
    load_azd_env(quiet=True)
    load_dotenv(".env")
    
    if not teacher_deployment_name:
        teacher_deployment_name = os.getenv("TEACHER_DEPLOYMENT_NAME")
    if not embedding_deployment_name:
        embedding_deployment_name = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    
    if not teacher_deployment_name:
        raise ValueError("TEACHER_DEPLOYMENT_NAME not found in environment or parameters")
    if not embedding_deployment_name:
        raise ValueError("EMBEDDING_DEPLOYMENT_NAME not found in environment or parameters")
    
    # Calculate QA threshold based on training split
    from math import ceil
    qa_threshold = ceil(finetuning_threshold / finetuning_train_split)
    
    ds_path = f"dataset/{ds_name}"
    
    # Create dataset directory
    os.makedirs(ds_path, exist_ok=True)
    
    return run_raft_generation(
        datapath=doc_path,
        output=ds_path,
        distractors=3,
        doctype="pdf",
        chunk_size=512,
        questions=raft_questions,
        workers=2,
        system_prompt_key="gpt",
        completion_model=teacher_deployment_name,
        embedding_model=embedding_deployment_name,
        qa_threshold=qa_threshold,
        completion_env_prefix="TEACHER"
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="RAFT Wrapper Example")
    parser.add_argument("--ds-name", default="zava-articles", help="Dataset name")
    parser.add_argument("--doc-path", default="sample_data/zava-articles", help="Document path")
    parser.add_argument("--questions", type=int, default=2, help="Questions per chunk")
    
    args = parser.parse_args()
    
    try:
        result = run_raft_from_notebook_params(
            ds_name=args.ds_name,
            doc_path=args.doc_path,
            raft_questions=args.questions
        )
        print("RAFT generation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
