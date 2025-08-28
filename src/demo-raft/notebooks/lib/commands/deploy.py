"""
RAFT Deploy Command

Command for deploying fine-tuned models to Azure OpenAI.
"""

import json
import logging
import os
import time
from typing import Optional

import requests
import rich_click as click
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from lib.shared import setup_environment, console, logger, create_azure_openai_client
from utils import update_state, get_env_state_file


def get_openai_client() -> AzureOpenAI:
    """Create and configure Azure OpenAI client."""
    return create_azure_openai_client()


def monitor_finetuning_job(client: AzureOpenAI, job_id: str, verbose: bool = False) -> str:
    """
    Monitor fine-tuning job until completion and return the fine-tuned model name.
    
    Args:
        client: Azure OpenAI client
        job_id: Fine-tuning job ID
        verbose: Enable verbose logging
        
    Returns:
        Fine-tuned model name
    """
    logger.info(f"🔍 Monitoring fine-tuning job: {job_id}")
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("Waiting for fine-tuning job completion...", total=None)
        
        while True:
            try:
                response = client.fine_tuning.jobs.retrieve(job_id)
                status = response.status
                
                if verbose:
                    progress.console.print(f"[dim]Job status: {status}[/dim]")
                
                if status in ["succeeded", "failed", "cancelled"]:
                    break
                    
                progress.update(
                    task,
                    description=f"Job status: [yellow]{status}[/yellow] - Elapsed: {{elapsed}}"
                )
                
                time.sleep(10)  # Poll every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Error checking job status: {e}")
                raise click.ClickException(f"Failed to monitor job: {e}")
    
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    
    if status == "succeeded":
        fine_tuned_model = response.fine_tuned_model
        logger.info(f"✅ Fine-tuning completed successfully in {minutes}m {seconds}s")
        logger.info(f"🎯 Fine-tuned model: {fine_tuned_model}")
        return fine_tuned_model
    elif status == "failed":
        logger.error(f"❌ Fine-tuning job failed after {minutes}m {seconds}s")
        if hasattr(response, 'error') and response.error:
            logger.error(f"   Error: {response.error}")
        raise click.ClickException("Fine-tuning job failed")
    else:
        logger.error(f"❌ Fine-tuning job was cancelled after {minutes}m {seconds}s")
        raise click.ClickException("Fine-tuning job was cancelled")


def create_model_deployment(
    fine_tuned_model: str,
    deployment_name: str,
    capacity: int = 4,
    verbose: bool = False
) -> dict:
    """
    Create a deployment for the fine-tuned model.
    
    Args:
        fine_tuned_model: Name of the fine-tuned model
        deployment_name: Name for the deployment
        capacity: Deployment capacity
        verbose: Enable verbose logging
        
    Returns:
        Deployment response dictionary
    """
    logger.info(f"🚀 Creating deployment: {deployment_name}")
    
    # Get Azure credentials and subscription info
    azure_credential = DefaultAzureCredential()
    access_token = azure_credential.get_token("https://management.azure.com/.default")
    
    subscription = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    aoai_endpoint = os.getenv("FINETUNE_AZURE_OPENAI_ENDPOINT")
    
    if not all([subscription, resource_group, aoai_endpoint]):
        missing = []
        if not subscription: missing.append("AZURE_SUBSCRIPTION_ID")
        if not resource_group: missing.append("AZURE_RESOURCE_GROUP")
        if not aoai_endpoint: missing.append("FINETUNE_AZURE_OPENAI_ENDPOINT")
        raise click.ClickException(f"❌ Missing environment variables: {', '.join(missing)}")
    
    resource_name = aoai_endpoint.split("https://")[1].split(".")[0]
    
    # Prepare deployment request
    deploy_params = {'api-version': "2023-05-01"}
    deploy_headers = {
        'Authorization': f'Bearer {access_token.token}',
        'Content-Type': 'application/json'
    }
    
    deploy_data = {
        "sku": {"name": "developertier", "capacity": capacity},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": fine_tuned_model,
                "version": "1"
            }
        }
    }
    
    request_url = (
        f'https://management.azure.com/subscriptions/{subscription}'
        f'/resourceGroups/{resource_group}'
        f'/providers/Microsoft.CognitiveServices/accounts/{resource_name}'
        f'/deployments/{deployment_name}'
    )
    
    if verbose:
        logger.debug(f"🔧 Deployment URL: {request_url}")
        logger.debug(f"🔧 Deployment data: {json.dumps(deploy_data, indent=2)}")
    
    try:
        response = requests.put(
            request_url,
            params=deploy_params,
            headers=deploy_headers,
            data=json.dumps(deploy_data)
        )
        
        if response.status_code not in [200, 201]:
            logger.error(f"❌ Deployment request failed: {response.status_code} {response.reason}")
            logger.error(f"   Response: {response.text}")
            raise click.ClickException(f"Failed to create deployment: {response.reason}")
        
        response_data = response.json()
        logger.info(f"✅ Deployment request submitted successfully")
        
        return {
            'url': request_url,
            'params': deploy_params,
            'headers': deploy_headers,
            'response': response_data
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error during deployment: {e}")
        raise click.ClickException(f"Failed to create deployment: {e}")


def monitor_deployment_status(
    request_url: str,
    deploy_params: dict,
    deploy_headers: dict,
    deployment_name: str,
    verbose: bool = False
) -> str:
    """
    Monitor deployment status until completion.
    
    Args:
        request_url: Deployment API URL
        deploy_params: Request parameters
        deploy_headers: Request headers
        deployment_name: Name of the deployment
        verbose: Enable verbose logging
        
    Returns:
        Final deployment status
    """
    logger.info(f"⏳ Monitoring deployment status for: {deployment_name}")
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("Waiting for deployment to complete...", total=None)
        
        while True:
            try:
                response = requests.get(request_url, params=deploy_params, headers=deploy_headers)
                
                if response.status_code != 200:
                    logger.error(f"❌ Failed to check deployment status: {response.status_code}")
                    break
                
                response_data = response.json()
                status = response_data.get('properties', {}).get('provisioningState', 'Unknown')
                
                if verbose:
                    progress.console.print(f"[dim]Deployment status: {status}[/dim]")
                
                if status.lower() in ["succeeded", "failed", "cancelled"]:
                    break
                    
                progress.update(
                    task,
                    description=f"Status: [yellow]{status}[/yellow] - Elapsed: {{elapsed}}"
                )
                
                time.sleep(5)  # Poll every 5 seconds for deployments
                
            except Exception as e:
                logger.error(f"❌ Error checking deployment status: {e}")
                break
    
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    
    if status.lower() == "succeeded":
        logger.info(f"✅ Deployment completed successfully in {minutes}m {seconds}s")
    else:
        logger.error(f"❌ Deployment finished with status: {status} after {minutes}m {seconds}s")
    
    return status


def display_deployment_summary(deployment_name: str, model_name: str, endpoint: str):
    """Display a summary of the deployment."""
    table = Table(title="🚀 Deployment Summary", show_header=True, header_style="bold blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Deployment Name", deployment_name)
    table.add_row("Fine-tuned Model", model_name)
    table.add_row("Azure OpenAI Endpoint", endpoint)
    table.add_row("Status", "Active")
    
    console.print(table)


@click.command()
@click.option("--job-id", help="Fine-tuning job ID (defaults from state)")
@click.option("--deployment-name", help="Custom deployment name (auto-generated if not provided)")
@click.option("--capacity", default=4, help="Deployment capacity (TPM quota)")
@click.option("--skip-monitoring", is_flag=True, help="Skip fine-tuning job monitoring")
@click.option("--model-name", help="Fine-tuned model name (if job monitoring is skipped)")
@click.option("--verbose", "-v", is_flag=True, help="Enable detailed logging output")
def deploy(
    job_id: Optional[str],
    deployment_name: Optional[str],
    capacity: int,
    skip_monitoring: bool,
    model_name: Optional[str],
    verbose: bool
):
    """
    Deploy fine-tuned models to [bold green]Azure OpenAI[/bold green].
    
    This command monitors fine-tuning job completion and creates a deployment
    for the resulting model. The deployed model can then be used for inference
    through the Azure OpenAI API.
    
    [bold yellow]Process Overview:[/bold yellow]
    [dim]1.[/dim] Monitor fine-tuning job until completion
    [dim]2.[/dim] Extract fine-tuned model name from job
    [dim]3.[/dim] Create deployment with specified capacity
    [dim]4.[/dim] Monitor deployment status until active
    
    [bold green]Example:[/bold green]
    [cyan]raft deploy --capacity 8 --verbose[/cyan]
    """
    # Configure logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 Verbose logging enabled")
    
    console.print("\n🚀 [bold blue]RAFT Model Deployment[/bold blue]\n")
    
    try:
        # Setup environment
        setup_environment()
        
        # Get job ID from state or parameter
        if not job_id:
            job_id = os.getenv("STUDENT_OPENAI_JOB_ID")
            if not job_id:
                raise click.ClickException("❌ No fine-tuning job ID found. Provide --job-id or run 'raft finetune' first.")
        
        logger.info(f"📋 Using fine-tuning job ID: {job_id}")
        
        # Get dataset name and model info for naming
        ds_name = os.getenv("DATASET_NAME", "default")
        student_model_name = os.getenv("STUDENT_MODEL_NAME", "gpt-4o-mini")
        
        # Generate deployment name if not provided
        if not deployment_name:
            deployment_name = f"ft-raft-{student_model_name}-{ds_name}"
        
        logger.info(f"🎯 Deployment name: {deployment_name}")
        
        # Get fine-tuned model name
        if skip_monitoring:
            if not model_name:
                raise click.ClickException("❌ --model-name is required when --skip-monitoring is used")
            fine_tuned_model = model_name
            logger.info(f"⏭️  Skipping job monitoring, using model: {fine_tuned_model}")
        else:
            # Create OpenAI client and monitor job
            client = get_openai_client()
            fine_tuned_model = monitor_finetuning_job(client, job_id, verbose)
        
        # Create deployment
        deployment_info = create_model_deployment(
            fine_tuned_model, deployment_name, capacity, verbose
        )
        
        # Monitor deployment status
        final_status = monitor_deployment_status(
            deployment_info['url'],
            deployment_info['params'],
            deployment_info['headers'],
            deployment_name,
            verbose
        )
        
        if final_status.lower() != "succeeded":
            raise click.ClickException(f"❌ Deployment failed with status: {final_status}")
        
        # Update state with deployment information
        logger.info("💾 Updating state with deployment information")
        update_state("STUDENT_DEPLOYMENT_NAME", deployment_name)
        update_state("STUDENT_AZURE_OPENAI_ENDPOINT", os.getenv("FINETUNE_AZURE_OPENAI_ENDPOINT"))
        update_state("STUDENT_AZURE_OPENAI_DEPLOYMENT", deployment_name)
        update_state("FINE_TUNED_MODEL_NAME", fine_tuned_model)
        
        # Display summary
        console.print("\n✅ [bold green]Deployment Created Successfully![/bold green]\n")
        display_deployment_summary(
            deployment_name, 
            fine_tuned_model, 
            os.getenv("FINETUNE_AZURE_OPENAI_ENDPOINT")
        )
        
        console.print(f"\n🎯 [bold]Next Steps:[/bold]")
        console.print("   • Your fine-tuned model is now deployed and ready for inference")
        console.print("   • Test the model using the Azure OpenAI Studio or API")
        console.print("   • Use the deployment for evaluation with 'raft eval'")
        console.print(f"   • Deployment name: [cyan]{deployment_name}[/cyan]")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if verbose:
            console.print_exception()
        raise click.ClickException(f"Deployment failed: {e}")
