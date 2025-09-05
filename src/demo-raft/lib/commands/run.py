"""
RAFT Run Command

Orchestrates the complete RAFT workflow by running multiple subcommands in sequence.
"""

import logging
import sys
from typing import Optional

import rich_click as click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from lib.shared import console, logger
from lib.commands.check import check
from lib.commands.gen import gen
from lib.commands.deploy import deploy
from lib.commands.eval import eval as eval_cmd
from lib.commands.finetune import finetune


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
              help="Citation format: legacy-xml-tag (no reformatting) or md-dash-list (reformat)")
@click.option("--skip-setup", is_flag=True, help="Skip RAFT repository setup in gen phase")
@click.option("--skip-check", is_flag=True, help="Skip infrastructure endpoint verification")
@click.option("--skip-gen", is_flag=True, help="Skip dataset generation (use existing dataset)")
@click.option("--skip-finetune", is_flag=True, help="Skip model fine-tuning step")
@click.option("--skip-deploy", is_flag=True, help="Skip model deployment")
@click.option("--skip-eval", is_flag=True, help="Skip model evaluation")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
def run(
    ds_name: str,
    doc_path: str,
    format_type: str,
    train_split: float,
    valid_split: float,
    finetuning_threshold: int,
    raft_questions: int,
    citation_format: str,
    skip_setup: bool,
    skip_check: bool,
    skip_gen: bool,
    skip_finetune: bool,
    skip_deploy: bool,
    skip_eval: bool,
    verbose: bool
):
    """
    Run the complete RAFT workflow [cyan]check[/cyan] → [cyan]gen[/cyan] → [cyan]finetune[/cyan] → [cyan]deploy[/cyan] → [cyan]eval[/cyan].

    This command orchestrates the entire RAFT process by executing multiple
    subcommands in sequence. It provides a streamlined way to go from raw
    documents to a fine-tuned, deployed, and evaluated model.
    
    [bold yellow]Workflow Steps:[/bold yellow]
    [dim]1.[/dim] [cyan]check[/cyan] - Verify Azure AI endpoints are accessible
    [dim]2.[/dim] [cyan]gen[/cyan] - Generate synthetic training dataset from documents
    [dim]3.[/dim] [cyan]finetune[/cyan] - Fine-tune a student model using generated dataset
    [dim]4.[/dim] [cyan]deploy[/cyan] - Deploy and fine-tune the model
    [dim]5.[/dim] [cyan]eval[/cyan] - Evaluate the fine-tuned model performance
    
    [bold yellow]Skip Options:[/bold yellow]
    Use [cyan]--skip-*[/cyan] flags to skip specific steps if you want to resume
    from a particular point in the workflow or if some steps are already complete.
    
    [bold green]Example:[/bold green]
    [cyan]raft run --ds-name my-dataset --raft-questions 3[/cyan]
    [cyan]raft run --skip-check --skip-gen  # Skip to finetune, deploy & eval[/cyan]
    """
    # Configure logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 Verbose logging enabled")
    
    console.print("\n🚀 [bold blue]RAFT Complete Workflow[/bold blue]\n")
    
    # Create workflow progress table
    workflow_table = Table(title="Workflow Steps")
    workflow_table.add_column("Step", style="cyan")
    workflow_table.add_column("Command", style="white")
    workflow_table.add_column("Status", style="yellow")
    
    workflow_table.add_row("1", "check", "⏭️ Skipped" if skip_check else "⏳ Pending")
    workflow_table.add_row("2", "gen", "⏭️ Skipped" if skip_gen else "⏳ Pending")
    workflow_table.add_row("3", "finetune", "⏭️ Skipped" if skip_finetune else "⏳ Pending")
    workflow_table.add_row("4", "deploy", "⏭️ Skipped" if skip_deploy else "⏳ Pending")
    workflow_table.add_row("5", "eval", "⏭️ Skipped" if skip_eval else "⏳ Pending")
    
    console.print(workflow_table)
    console.print()
    
    try:
        step_count = 0
        completed_steps = 0
        total_steps = sum(1 for skip in [skip_check, skip_gen, skip_finetune, skip_deploy, skip_eval] if not skip)
        
        # Step 1: Check infrastructure
        if not skip_check:
            step_count += 1
            console.print(f"📍 [bold]Step {step_count}/{total_steps}: Infrastructure Check[/bold]")
            
            # Create a new Click context for the check command
            ctx = click.Context(check)
            ctx.params = {'verbose': verbose}
            
            try:
                ctx.invoke(check, verbose=verbose)
                completed_steps += 1
                console.print("✅ Infrastructure check completed successfully\n")
            except Exception as e:
                logger.error(f"❌ Infrastructure check failed: {e}")
                raise click.ClickException("Infrastructure check failed. Please fix the issues before continuing.")
        
        # Step 2: Generate dataset
        if not skip_gen:
            step_count += 1
            console.print(f"📍 [bold]Step {step_count}/{total_steps}: Dataset Generation[/bold]")
            
            # Create a new Click context for the gen command
            ctx = click.Context(gen)
            ctx.params = {
                'ds_name': ds_name,
                'doc_path': doc_path,
                'format_type': format_type,
                'train_split': train_split,
                'valid_split': valid_split,
                'finetuning_threshold': finetuning_threshold,
                'raft_questions': raft_questions,
                'citation_format': citation_format,
                'skip_setup': skip_setup,
                'verbose': verbose
            }
            
            try:
                ctx.invoke(gen, 
                          ds_name=ds_name,
                          doc_path=doc_path,
                          format_type=format_type,
                          train_split=train_split,
                          valid_split=valid_split,
                          finetuning_threshold=finetuning_threshold,
                          raft_questions=raft_questions,
                          citation_format=citation_format,
                          skip_setup=skip_setup,
                          verbose=verbose)
                completed_steps += 1
                console.print("✅ Dataset generation completed successfully\n")
            except Exception as e:
                logger.error(f"❌ Dataset generation failed: {e}")
                raise click.ClickException("Dataset generation failed. Please fix the issues before continuing.")
        
        # Step 3: Fine-tune model
        if not skip_finetune:
            step_count += 1
            console.print(f"📍 [bold]Step {step_count}/{total_steps}: Model Fine-tuning[/bold]")

            # Create a new Click context for the finetune command
            ctx = click.Context(finetune)
            ctx.params = {'verbose': verbose}

            try:
                ctx.invoke(finetune, verbose=verbose)
                completed_steps += 1
                console.print("✅ Fine-tuning step completed successfully\n")
            except Exception as e:
                logger.error(f"❌ Fine-tuning failed: {e}")
                raise click.ClickException("Fine-tuning failed. Please fix the issues before continuing.")
        
        # Step 4: Deploy model
        if not skip_deploy:
            step_count += 1
            console.print(f"📍 [bold]Step {step_count}/{total_steps}: Model Deployment[/bold]")
            
            # Create a new Click context for the deploy command
            ctx = click.Context(deploy)
            ctx.params = {'verbose': verbose}
            
            try:
                ctx.invoke(deploy, verbose=verbose)
                completed_steps += 1
                console.print("✅ Model deployment completed successfully\n")
            except Exception as e:
                logger.error(f"❌ Model deployment failed: {e}")
                raise click.ClickException("Model deployment failed. Please fix the issues before continuing.")
        
        # Step 5: Evaluate model
        if not skip_eval:
            step_count += 1
            console.print(f"📍 [bold]Step {step_count}/{total_steps}: Model Evaluation[/bold]")
            
            # Create a new Click context for the eval command
            ctx = click.Context(eval_cmd)
            ctx.params = {'verbose': verbose}
            
            try:
                ctx.invoke(eval_cmd, verbose=verbose)
                completed_steps += 1
                console.print("✅ Model evaluation completed successfully\n")
            except Exception as e:
                logger.error(f"❌ Model evaluation failed: {e}")
                raise click.ClickException("Model evaluation failed. Please fix the issues before continuing.")
        
        # Success summary
        console.print("\n🎉 [bold green]RAFT Workflow Complete![/bold green]\n")
        
        summary_table = Table(title="Workflow Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Steps", str(total_steps))
        summary_table.add_row("Completed Steps", str(completed_steps))
        summary_table.add_row("Success Rate", f"{(completed_steps/total_steps)*100:.1f}%" if total_steps > 0 else "N/A")
        summary_table.add_row("Dataset Name", ds_name)
        
        console.print(summary_table)
        
        console.print(f"\n🎯 [bold]Your RAFT workflow is complete![/bold]")
        console.print("   • Check the logs above for detailed results")
        console.print("   • Your fine-tuned model should now be deployed and evaluated")
        console.print("   • Use [cyan]raft status[/cyan] to check current deployments")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in workflow: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)
