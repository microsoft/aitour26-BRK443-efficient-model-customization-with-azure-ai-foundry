"""
RAFT Evaluation Command

Command for evaluating fine-tuned models against baseline models using synthetic datasets.
"""

import json
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import rich_click as click
import pandas as pd
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console

from lib.shared import setup_environment, console, logger, execute_command
from utils import get_env_state_file, update_state


def setup_eval_environment() -> Dict[str, str]:
    """
    Set up environment variables for evaluation and return model configurations.
    
    Returns:
        Dictionary containing model configurations
    """
    setup_environment()
    
    # Required environment variables
    required_vars = {
        'BASELINE_DEPLOYMENT_NAME': 'Baseline model deployment name',
        'BASELINE_MODEL_API': 'Baseline model API type',
        'STUDENT_DEPLOYMENT_NAME': 'Student model deployment name', 
        'STUDENT_MODEL_API': 'Student model API type',
        'DATASET_NAME': 'Dataset name for evaluation'
    }
    
    config = {}
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"{var} ({description})")
        else:
            config[var] = value
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   • {var}")
        raise click.ClickException(f"Missing {len(missing_vars)} required environment variable(s)")
    
    logger.info(f"✅ Environment configured for evaluation")
    logger.info(f"   • Baseline: {config['BASELINE_MODEL_API']} model {config['BASELINE_DEPLOYMENT_NAME']}")
    logger.info(f"   • Student: {config['STUDENT_MODEL_API']} model {config['STUDENT_DEPLOYMENT_NAME']}")
    logger.info(f"   • Dataset: {config['DATASET_NAME']}")
    
    return config


def create_experiment_paths(dataset_name: str) -> Dict[str, str]:
    """
    Create and return all file paths needed for evaluation.
    
    Args:
        dataset_name: Name of the dataset to evaluate
        
    Returns:
        Dictionary of file paths
    """
    cwd = os.getcwd()
    experiment_dir = f"{cwd}/dataset/{dataset_name}-files"
    
    paths = {
        'experiment_dir': experiment_dir,
        'dataset_eval': f"{experiment_dir}/{dataset_name}-hf.eval.jsonl",
        'baseline_answers': f"{experiment_dir}/{dataset_name}-hf.eval.answer.baseline.jsonl",
        'student_answers': f"{experiment_dir}/{dataset_name}-hf.eval.answer.jsonl",
        'baseline_formatted': f"{experiment_dir}/{dataset_name}-eval.answer.baseline.jsonl",
        'student_formatted': f"{experiment_dir}/{dataset_name}-eval.answer.student.jsonl",
        'baseline_scores': f"{experiment_dir}/{dataset_name}-eval.answer.score.baseline.jsonl",
        'student_scores': f"{experiment_dir}/{dataset_name}-eval.answer.score.student.jsonl",
        'baseline_metrics': f"{experiment_dir}/{dataset_name}-eval.answer.score.metrics.baseline.json",
        'student_metrics': f"{experiment_dir}/{dataset_name}-eval.answer.score.metrics.student.json"
    }
    
    return paths


def run_model_evaluation(
    eval_dataset_path: str,
    output_path: str, 
    model_deployment: str,
    env_prefix: str,
    model_api: str,
    verbose: bool = False
) -> bool:
    """
    Run model evaluation using the RAFT evaluation script.
    
    Args:
        eval_dataset_path: Path to evaluation dataset
        output_path: Path to save answers
        model_deployment: Model deployment name
        env_prefix: Environment variable prefix (BASELINE or STUDENT)
        model_api: Model API type
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    if os.path.exists(output_path):
        logger.info(f"📄 Answers file already exists: {output_path}")
        return True
    
    # Get environment files
    env_state_file = get_env_state_file()
    azure_env_name = os.getenv("AZURE_ENV_NAME", "")
    azure_env_file = f".azure/{azure_env_name}/.env" if azure_env_name else ""
    
    # Build environment command
    env_files = [f for f in [".env", env_state_file, azure_env_file] if f and os.path.exists(f)]
    env_command = f"env $(cat {' '.join(env_files)})" if env_files else ""
    
    # Build evaluation command
    eval_command = [
        "python", ".gorilla/raft/eval.py",
        "--question-file", eval_dataset_path,
        "--answer-file", output_path,
        "--model", model_deployment,
        "--env-prefix", env_prefix,
        "--mode", model_api
    ]
    
    command = f"{env_command} {' '.join(eval_command)}"
    
    logger.info(f"🔄 Running {env_prefix.lower()} model evaluation")
    if verbose:
        logger.debug(f"Command: {command}")
    
    return_code, stdout, stderr = execute_command(
        command,
        description=f"Evaluating {env_prefix.lower()} model"
    )
    
    if return_code != 0:
        logger.error(f"❌ Model evaluation failed: {stderr}")
        return False
    
    logger.info(f"✅ {env_prefix.lower().title()} model evaluation completed")
    return True


def format_answers(input_path: str, output_path: str, verbose: bool = False) -> bool:
    """
    Format model answers for evaluation using the RAFT format script.
    
    Args:
        input_path: Path to raw answers file
        output_path: Path to save formatted answers
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python", ".gorilla/raft/format.py",
        "--input", input_path,
        "--input-type", "jsonl", 
        "--output", output_path,
        "--output-format", "eval"
    ]
    
    if verbose:
        logger.debug(f"Format command: {' '.join(command)}")
    
    return_code, stdout, stderr = execute_command(
        ' '.join(command),
        description="Formatting answers for evaluation"
    )
    
    if return_code != 0:
        logger.error(f"❌ Answer formatting failed: {stderr}")
        return False
    
    logger.info("✅ Answer formatting completed")
    return True


def setup_evaluation_config() -> Dict[str, Any]:
    """
    Set up evaluation configuration including judge model.
    
    Returns:
        Evaluation configuration dictionary
    """
    from azure.ai.evaluation import OpenAIModelConfiguration, AzureOpenAIModelConfiguration
    
    # Check for OpenAI configuration
    openai_base_url = os.environ.get("JUDGE_OPENAI_BASE_URL")
    azure_endpoint = os.environ.get("JUDGE_AZURE_OPENAI_ENDPOINT")
    
    if openai_base_url:
        openai_api_key = os.environ.get("JUDGE_OPENAI_API_KEY")
        model = os.environ.get("JUDGE_OPENAI_DEPLOYMENT")
        
        logger.info(f"🔧 Using OpenAI judge model: {model}")
        
        model_config = OpenAIModelConfiguration(
            base_url=openai_base_url,
            api_key=openai_api_key,
            model=model
        )
        model_config.api_version = None
        
    elif azure_endpoint:
        azure_deployment = os.environ.get("JUDGE_AZURE_OPENAI_DEPLOYMENT")
        api_key = os.environ.get("JUDGE_AZURE_OPENAI_API_KEY") 
        api_version = os.environ.get("JUDGE_OPENAI_API_VERSION")
        
        logger.info(f"🔧 Using Azure OpenAI judge model: {azure_deployment}")
        
        args = {
            'azure_endpoint': azure_endpoint,
            'azure_deployment': azure_deployment,
            'api_version': api_version,
        }
        if api_key:
            args['api_key'] = api_key
            
        model_config = AzureOpenAIModelConfiguration(**args)
        
    else:
        raise click.ClickException("❌ No judge model endpoint found. Set JUDGE_OPENAI_BASE_URL or JUDGE_AZURE_OPENAI_ENDPOINT")
    
    return {'model_config': model_config}


def create_evaluators(model_config: Any) -> Dict[str, Any]:
    """
    Create and return evaluation metrics.
    
    Args:
        model_config: Model configuration for GPT-based evaluators
        
    Returns:
        Dictionary of evaluators
    """
    from azure.ai.evaluation import (
        CoherenceEvaluator, F1ScoreEvaluator, FluencyEvaluator, 
        GroundednessEvaluator, RelevanceEvaluator, SimilarityEvaluator,
        BleuScoreEvaluator, RougeScoreEvaluator, RougeType
    )
    
    evaluators = {
        # GPT-based metrics
        "coherence": CoherenceEvaluator(model_config),
        "fluency": FluencyEvaluator(model_config),
        "groundedness": GroundednessEvaluator(model_config),
        "relevance": RelevanceEvaluator(model_config),
        "similarity": SimilarityEvaluator(model_config),
        
        # Math metrics
        "f1_score": F1ScoreEvaluator(),
        "bleu": BleuScoreEvaluator(),
        "rouge_1": RougeScoreEvaluator(RougeType.ROUGE_1),
        "rouge_2": RougeScoreEvaluator(RougeType.ROUGE_2),
    }
    
    logger.info(f"✅ Created {len(evaluators)} evaluation metrics")
    return evaluators


def score_dataset(
    dataset_path: str, 
    evaluators: Dict[str, Any], 
    rows_output_path: Optional[str] = None,
    metrics_output_path: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Score a dataset using the evaluation API.
    
    Args:
        dataset_path: Path to formatted dataset file
        evaluators: Dictionary of evaluators
        rows_output_path: Path to save detailed row results
        metrics_output_path: Path to save aggregated metrics
        verbose: Enable verbose logging
        
    Returns:
        Evaluation results dictionary
    """
    from azure.ai.evaluation import evaluate
    
    logger.info(f"🔄 Scoring dataset: {dataset_path}")
    
    if verbose:
        logger.debug(f"Using {len(evaluators)} evaluators")
    
    result = evaluate(
        data=dataset_path,
        evaluators=evaluators,
        evaluator_config={
            "default": {
                "column_mapping": {
                    "query": "${data.question}",
                    "response": "${data.final_answer}",
                    "ground_truth": "${data.gold_final_answer}",
                    "context": "${data.context}",
                }
            }
        },
    )
    
    # Save detailed results if requested
    if rows_output_path:
        pd.DataFrame.from_dict(result["rows"]).to_json(
            rows_output_path, orient="records", lines=True
        )
        logger.info(f"💾 Detailed results saved to: {rows_output_path}")
    
    # Save metrics if requested  
    if metrics_output_path:
        with open(metrics_output_path, "w") as f:
            json.dump(result['metrics'], f, indent=2)
        logger.info(f"💾 Metrics saved to: {metrics_output_path}")
    
    logger.info("✅ Dataset scoring completed")
    return result


def display_metrics_comparison(
    baseline_metrics_path: str,
    student_metrics_path: str,
    verbose: bool = False
):
    """
    Display a comparison of baseline vs student metrics.
    
    Args:
        baseline_metrics_path: Path to baseline metrics JSON
        student_metrics_path: Path to student metrics JSON
        verbose: Enable verbose logging
    """
    # Load metrics
    with open(baseline_metrics_path, "r") as f:
        baseline_metrics = json.load(f)
    
    with open(student_metrics_path, "r") as f:
        student_metrics = json.load(f)
    
    # Create comparison table
    table = Table(title="🎯 Model Performance Comparison", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="yellow", justify="right")
    table.add_column("Student", style="green", justify="right") 
    table.add_column("Improvement", style="magenta", justify="right")
    
    # Calculate improvements and add rows
    for metric_name in sorted(baseline_metrics.keys()):
        baseline_val = baseline_metrics[metric_name]
        student_val = student_metrics.get(metric_name, 0)
        
        if baseline_val != 0:
            improvement = ((student_val - baseline_val) / baseline_val) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        table.add_row(
            metric_name,
            f"{baseline_val:.3f}",
            f"{student_val:.3f}",
            improvement_str
        )
    
    console.print("\n")
    console.print(table)
    
    if verbose:
        logger.debug(f"Baseline metrics: {baseline_metrics}")
        logger.debug(f"Student metrics: {student_metrics}")


@click.command()
@click.option("--dataset-name", help="Dataset name to evaluate (defaults from state)")
@click.option("--skip-baseline", is_flag=True, help="Skip baseline model evaluation")
@click.option("--skip-student", is_flag=True, help="Skip student model evaluation") 
@click.option("--force-regenerate", is_flag=True, help="Force regenerate all evaluation files")
@click.option("--verbose", "-v", is_flag=True, help="Enable detailed logging output")
def eval(
    dataset_name: Optional[str],
    skip_baseline: bool,
    skip_student: bool,
    force_regenerate: bool,
    verbose: bool
):
    """
    Evaluate fine-tuned models against [bold green]baseline models[/bold green].
    
    This command runs both baseline and fine-tuned student models on the evaluation
    dataset, then compares their performance using multiple metrics including
    GPT-based quality evaluators and mathematical similarity measures.
    
    [bold yellow]Process Overview:[/bold yellow]
    [dim]1.[/dim] Run baseline model on evaluation dataset
    [dim]2.[/dim] Run student model on evaluation dataset  
    [dim]3.[/dim] Format answers for evaluation framework
    [dim]4.[/dim] Score both models using multiple metrics
    [dim]5.[/dim] Display performance comparison
    
    [bold green]Example:[/bold green]
    [cyan]raft eval --verbose[/cyan]
    """
    # Configure logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 Verbose logging enabled")
    
    console.print("\n📊 [bold blue]RAFT Model Evaluation[/bold blue]\n")
    
    try:
        # Setup environment
        config = setup_eval_environment()
        
        # Get dataset name
        if not dataset_name:
            dataset_name = config.get('DATASET_NAME')
            if not dataset_name:
                raise click.ClickException("❌ No dataset name provided. Use --dataset-name or set DATASET_NAME")
        
        logger.info(f"🎯 Evaluating dataset: {dataset_name}")
        
        # Create file paths
        paths = create_experiment_paths(dataset_name)
        
        # Check if evaluation dataset exists
        if not os.path.exists(paths['dataset_eval']):
            raise click.ClickException(f"❌ Evaluation dataset not found: {paths['dataset_eval']}")
        
        # Phase 1: Run model evaluations
        console.print("🔄 [bold]Phase 1: Running Model Evaluations[/bold]")
        
        if not skip_baseline:
            if force_regenerate and os.path.exists(paths['baseline_answers']):
                os.remove(paths['baseline_answers'])
            
            success = run_model_evaluation(
                paths['dataset_eval'],
                paths['baseline_answers'],
                config['BASELINE_DEPLOYMENT_NAME'],
                'BASELINE',
                config['BASELINE_MODEL_API'],
                verbose
            )
            if not success:
                raise click.ClickException("❌ Baseline model evaluation failed")
        
        if not skip_student:
            if force_regenerate and os.path.exists(paths['student_answers']):
                os.remove(paths['student_answers'])
            
            success = run_model_evaluation(
                paths['dataset_eval'],
                paths['student_answers'], 
                config['STUDENT_DEPLOYMENT_NAME'],
                'STUDENT',
                config['STUDENT_MODEL_API'],
                verbose
            )
            if not success:
                raise click.ClickException("❌ Student model evaluation failed")
        
        # Phase 2: Format answers
        console.print("\n🔄 [bold]Phase 2: Formatting Answers[/bold]")
        
        if not skip_baseline:
            if force_regenerate and os.path.exists(paths['baseline_formatted']):
                os.remove(paths['baseline_formatted'])
            
            if not os.path.exists(paths['baseline_formatted']):
                success = format_answers(paths['baseline_answers'], paths['baseline_formatted'], verbose)
                if not success:
                    raise click.ClickException("❌ Baseline answer formatting failed")
        
        if not skip_student:
            if force_regenerate and os.path.exists(paths['student_formatted']):
                os.remove(paths['student_formatted'])
            
            if not os.path.exists(paths['student_formatted']):
                success = format_answers(paths['student_answers'], paths['student_formatted'], verbose)
                if not success:
                    raise click.ClickException("❌ Student answer formatting failed")
        
        # Phase 3: Score models
        console.print("\n🔄 [bold]Phase 3: Scoring Models[/bold]")
        
        # Setup evaluation framework
        eval_config = setup_evaluation_config()
        evaluators = create_evaluators(eval_config['model_config'])
        
        if not skip_baseline:
            if force_regenerate and os.path.exists(paths['baseline_scores']):
                os.remove(paths['baseline_scores'])
            if force_regenerate and os.path.exists(paths['baseline_metrics']):
                os.remove(paths['baseline_metrics'])
            
            if not os.path.exists(paths['baseline_metrics']):
                logger.info("🎯 Scoring baseline model")
                baseline_result = score_dataset(
                    paths['baseline_formatted'],
                    evaluators,
                    paths['baseline_scores'],
                    paths['baseline_metrics'],
                    verbose
                )
                
                studio_url = baseline_result.get("studio_url", "http://127.0.0.1:23333")
                logger.info(f"🌐 Baseline results available at: {studio_url}")
        
        if not skip_student:
            if force_regenerate and os.path.exists(paths['student_scores']):
                os.remove(paths['student_scores'])
            if force_regenerate and os.path.exists(paths['student_metrics']):
                os.remove(paths['student_metrics'])
            
            if not os.path.exists(paths['student_metrics']):
                logger.info("🎯 Scoring student model")
                student_result = score_dataset(
                    paths['student_formatted'],
                    evaluators,
                    paths['student_scores'], 
                    paths['student_metrics'],
                    verbose
                )
                
                studio_url = student_result.get("studio_url", "http://127.0.0.1:23333") 
                logger.info(f"🌐 Student results available at: {studio_url}")
        
        # Phase 4: Display comparison
        if not skip_baseline and not skip_student:
            if os.path.exists(paths['baseline_metrics']) and os.path.exists(paths['student_metrics']):
                console.print("\n🔄 [bold]Phase 4: Performance Comparison[/bold]")
                display_metrics_comparison(
                    paths['baseline_metrics'],
                    paths['student_metrics'],
                    verbose
                )
        
        # Update state
        logger.info("💾 Updating state with evaluation results")
        update_state("LAST_EVAL_DATASET", dataset_name)
        if not skip_baseline:
            update_state("BASELINE_EVAL_METRICS", paths['baseline_metrics'])
        if not skip_student:
            update_state("STUDENT_EVAL_METRICS", paths['student_metrics'])
        
        console.print("\n✅ [bold green]Evaluation Completed Successfully![/bold green]")
        
        # Display next steps
        console.print(f"\n🎯 [bold]Results Summary:[/bold]")
        console.print(f"   • Dataset evaluated: [cyan]{dataset_name}[/cyan]")
        if not skip_baseline:
            console.print(f"   • Baseline metrics: [yellow]{paths['baseline_metrics']}[/yellow]")
        if not skip_student:
            console.print(f"   • Student metrics: [green]{paths['student_metrics']}[/green]")
        
        console.print(f"\n🎯 [bold]Next Steps:[/bold]")
        console.print("   • Review detailed metrics in the generated JSON files")
        console.print("   • Use Azure AI Studio for detailed result analysis")
        console.print("   • Consider adjusting fine-tuning parameters based on results")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during evaluation: {e}")
        if verbose:
            console.print_exception()
        raise click.ClickException(f"Evaluation failed: {e}")
