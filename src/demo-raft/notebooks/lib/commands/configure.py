"""
RAFT Configure Command

Command for configuring AI models and deployments.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

import rich_click as click
import yaml
from dotenv import load_dotenv
from dotenv_azd import load_azd_env
from rich.table import Table

from lib.shared import setup_environment, console, logger


class Model:
    """Represents an AI model configuration."""
    
    def __init__(self, data: dict):
        self.data = data

    @property
    def name(self) -> str:
        return self.data['name']

    @property
    def api(self) -> str:
        return self.data['api']

    @property
    def version(self) -> str:
        return self.data['version']


class Descriptor:
    """Represents a deployment descriptor."""
    
    def __init__(self, data: dict):
        self.data = data

    def is_supported_in_regions(self, regions: Set[str]) -> bool:
        common = set(self.regions & set(regions))
        return len(common) > 0
    
    @property
    def regions(self) -> Set[str]:
        return set(self.data['regions'])

    @property
    def model(self) -> Model:
        return Model(self.data['model'])


class Descriptors:
    """Collection of deployment descriptors."""
    
    def __init__(self, ai_config):
        self.ai_config = ai_config

    def __getitem__(self, key: str) -> Descriptor:
        models = self.ai_config.data['deployments'] if 'deployments' in self.ai_config.data else []
        val = next(filter(lambda d: d['name'] == key, models), None)
        if not val:
            raise ValueError(f"Model {key} not found")
        return Descriptor(val)


class AiConfig:
    """AI configuration container."""
    
    def __init__(self, data: dict):
        self.data = data

    @property
    def descriptors(self) -> Descriptors:
        return Descriptors(self)


def read_ai_config() -> AiConfig:
    """Read AI configuration from ai.yaml file."""
    # Look for ai.yaml in infra/azd directory relative to current directory
    current_dir = Path.cwd()
    possible_paths = [
        current_dir / "infra" / "azd" / "ai.yaml",
        current_dir.parent / "infra" / "azd" / "ai.yaml",
        current_dir / ".." / "infra" / "azd" / "ai.yaml",
    ]
    
    ai_config_path = None
    for path in possible_paths:
        if path.exists():
            ai_config_path = path
            break
    
    if not ai_config_path:
        raise click.ClickException("❌ ai.yaml configuration file not found. Expected in infra/azd/ directory.")
    
    logger.debug(f"🔧 Reading AI config from: {ai_config_path}")
    
    with open(ai_config_path, 'r') as config_file:
        ai_config_data = yaml.safe_load(config_file)
    
    return AiConfig(ai_config_data)


def get_roles(ai_config_data: dict) -> List[str]:
    """Extract all unique roles from the AI configuration."""
    deployments = ai_config_data.get('deployments', [])
    roles = list(set([role for d in deployments for role in d['roles']]))
    roles.sort()
    return roles


def get_regions(ai_config_data: dict) -> Set[str]:
    """Extract all unique regions from the AI configuration."""
    deployments = ai_config_data.get('deployments', [])
    regions = set([region for d in deployments for region in d['regions']])
    return regions


def filter_provider_transfer_compliant_models(deployment: dict, role: str, selected_platforms: Dict[str, str]) -> bool:
    """
    Ensures platform consistency for teacher and student model selections.
    
    This function ensures compliance with model provider licensing constraints
    by restricting teacher/student model selections to the same platform.
    """
    if not selected_platforms:
        return True
    
    platform = deployment.get('platform')
    if not platform:
        return True
        
    # If this is a teacher/student role and a platform has been selected for any teacher/student role,
    # only include deployments with the same platform
    if role in ['teacher', 'student']:
        for selected_role, selected_platform in selected_platforms.items():
            if selected_role in ['teacher', 'student']:
                return platform == selected_platform
    
    return True


def get_finetuning_sku_name(ai_config_data: dict, deployment_name: str) -> Optional[str]:
    """Extract the finetuning SKU name for a specific deployment from the AI configuration."""
    deployments = ai_config_data.get('deployments', [])
    
    for deployment in deployments:
        if deployment.get('name') == deployment_name:
            finetuning_config = deployment.get('finetuning')
            if finetuning_config and isinstance(finetuning_config, list):
                for ft_config in finetuning_config:
                    sku_config = ft_config.get('sku')
                    if sku_config and isinstance(sku_config, list):
                        for sku in sku_config:
                            if isinstance(sku, dict) and 'name' in sku:
                                return sku['name']
    
    return None


def role_finetuning_sku_env_var_name(role: str) -> str:
    """Generate environment variable name for a role's finetuning SKU."""
    return f'{role.upper()}_FINETUNING_SKU_NAME'


def get_deployment_names(
    ai_config_data: dict, 
    regions: Set[str], 
    role: str = 'teacher', 
    selected_platforms: Optional[Dict[str, str]] = None
) -> List[str]:
    """Get deployment names for a specific role and region set."""
    deployments = ai_config_data.get('deployments', [])

    def deployment_filter(d):
        return (
            role in d['roles']
            and Descriptor(d).is_supported_in_regions(regions)
            and filter_provider_transfer_compliant_models(d, role, selected_platforms)
        )

    filtered_deployments = filter(deployment_filter, deployments)
    deployment_names = [d['name'] for d in filtered_deployments]
    return deployment_names


def select_model_interactive(role: str, names: List[str], default: Optional[str] = None) -> str:
    """Interactively select a model from available options."""
    try:
        import survey
    except ImportError:
        # Fallback if survey is not available - use the first option or default
        if default and default in names:
            console.print(f"⚠️  Interactive selection not available. Using default: [cyan]{default}[/cyan]")
            return default
        elif names:
            console.print(f"⚠️  Interactive selection not available. Using first option: [cyan]{names[0]}[/cyan]")
            return names[0]
        else:
            raise click.ClickException(f"❌ No {role} models available and no fallback option")
    
    default_index = names.index(default) if default and default in names else 0
    console.print(f"\n🎯 Select a [bold cyan]{role}[/bold cyan] deployment:")
    
    try:
        index = survey.routines.select(
            f"Pick a {role} deployment name: ", 
            options=names, 
            index=default_index
        )
        return names[index]
    except (KeyboardInterrupt, EOFError):
        console.print("\n⚠️  Selection cancelled, using default.")
        return names[default_index] if names else None


def select_region_interactive(regions: List[str], default: Optional[str] = None) -> str:
    """Interactively select a region from available options."""
    try:
        import survey
    except ImportError:
        # Fallback if survey is not available
        if default and default in regions:
            console.print(f"⚠️  Interactive selection not available. Using default region: [cyan]{default}[/cyan]")
            return default
        elif regions:
            console.print(f"⚠️  Interactive selection not available. Using first region: [cyan]{regions[0]}[/cyan]")
            return regions[0]
        else:
            raise click.ClickException("❌ No regions available")
    
    regions_list = list(regions)
    default_index = regions_list.index(default) if default and default in regions_list else 0
    console.print(f"\n🌍 Select a region:")
    
    # Display options
    for i, region in enumerate(regions_list):
        marker = "👉" if i == default_index else "   "
        console.print(f"   {marker} [{i}] {region}")
    
    try:
        index = survey.routines.select(
            "Pick a region: ", 
            options=regions_list, 
            index=default_index
        )
        return regions_list[index]
    except (KeyboardInterrupt, EOFError):
        console.print("\n⚠️  Selection cancelled, using default.")
        return regions_list[default_index] if regions_list else None


def azd_set_env(name: str, value: str) -> bool:
    """Set an environment variable using azd."""
    try:
        result = subprocess.run(
            ['azd', 'env', 'set', name, value], 
            shell=False, 
            capture_output=True, 
            text=True,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Failed to set azd environment variable {name}: {e}")
        return False
    except FileNotFoundError:
        logger.warning("⚠️  azd command not found. Skipping environment variable setting.")
        return False


def role_deployment_env_var_name(role: str) -> str:
    """Generate environment variable name for a role's deployment."""
    return f'{role.upper()}_DEPLOYMENT_NAME'


def role_model_env_var_name(role: str) -> str:
    """Generate environment variable name for a role's model."""
    return f'{role.upper()}_MODEL_NAME'


def role_model_api_env_var_name(role: str) -> str:
    """Generate environment variable name for a role's API."""
    return f'{role.upper()}_MODEL_API'


def display_configuration_summary(selections: List[tuple], region: str):
    """Display a summary table of the configuration."""
    table = Table(title="🎯 Configuration Summary", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Group by role for better readability
    roles_shown = set()
    azure_location = None
    
    for name, value in selections:
        if "_DEPLOYMENT_NAME" in name:
            role = name.replace("_DEPLOYMENT_NAME", "").lower()
            if role not in roles_shown:
                table.add_row(f"{role.title()} Role", "")
                roles_shown.add(role)
        
        # Handle special cases
        if name == "AZURE_LOCATION":
            azure_location = value
        else:
            table.add_row(f"  {name}", value)
    
    # Add region at the end
    if azure_location:
        table.add_row("Region", azure_location)
    
    console.print("\n")
    console.print(table)


@click.command()
@click.option('--set-azd-env/--no-set-azd-env', default=True, 
              help='Set selected deployment names as azd environment variables')
@click.option('--region', '-r', multiple=True, default=None, 
              help='Restrict which regions to consider for models (defaults to all regions)')
@click.option('--teacher-deployment', help='Teacher model deployment name')
@click.option('--student-deployment', help='Student model deployment name') 
@click.option('--judge-deployment', help='Judge model deployment name')
@click.option('--baseline-deployment', help='Baseline model deployment name')
@click.option('--embedding-deployment', help='Embedding model deployment name')
@click.option('--non-interactive', is_flag=True, help='Use defaults without interactive selection')
@click.option('--verbose', '-v', is_flag=True, help='Enable detailed logging output')
def configure(
    set_azd_env: bool,
    region: tuple,
    teacher_deployment: Optional[str],
    student_deployment: Optional[str],
    judge_deployment: Optional[str],
    baseline_deployment: Optional[str],
    embedding_deployment: Optional[str],
    non_interactive: bool,
    verbose: bool
):
    """
    Configure AI models and deployments for [bold green]RAFT workflows[/bold green].
    
    This command reads the AI configuration and allows selection of specific
    model deployments for different roles (teacher, student, judge, baseline, embedding).
    Ensures platform consistency to comply with licensing constraints.
    
    [bold yellow]Process Overview:[/bold yellow]
    [dim]1.[/dim] Read AI configuration from infra/azd/ai.yaml
    [dim]2.[/dim] Filter available deployments by region and role  
    [dim]3.[/dim] Select models interactively or use provided defaults
    [dim]4.[/dim] Set environment variables for selected models
    
    [bold green]Example:[/bold green]
    [cyan]raft configure --region eastus --set-azd-env[/cyan]
    """
    # Configure logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 Verbose logging enabled")
    
    console.print("\n⚙️  [bold blue]RAFT Model Configuration[/bold blue]\n")
    
    try:
        # Setup environment
        setup_environment()
        
        # Read AI configuration
        logger.info("📖 Reading AI configuration")
        ai_config = read_ai_config()
        roles = get_roles(ai_config.data)
        all_regions = get_regions(ai_config.data)
        
        logger.info(f"🎭 Available roles: {', '.join(roles)}")
        logger.info(f"🌍 Available regions: {', '.join(sorted(all_regions))}")
        
        # Filter regions if specified
        if region:
            regions = all_regions & set(region)
            if not regions:
                raise click.ClickException(f"❌ No valid regions found. Available: {', '.join(sorted(all_regions))}")
        else:
            regions = all_regions
        
        logger.info(f"🎯 Using regions: {', '.join(sorted(regions))}")
        
        # Build role to deployment mapping
        role_deployments = {
            'teacher': teacher_deployment,
            'student': student_deployment, 
            'judge': judge_deployment,
            'baseline': baseline_deployment,
            'embedding': embedding_deployment
        }
        
        # Filter to only roles that exist in the configuration
        role_deployments = {role: deployment for role, deployment in role_deployments.items() if role in roles}
        
        console.print(f"🎭 Configuring models for roles: [cyan]{', '.join(role_deployments.keys())}[/cyan]")
        console.print("📍 Each selection narrows down future selections based on the region.\n")
        
        selections = []
        selected_platforms = {}
        current_regions = regions.copy()
        
        for role in role_deployments.keys():
            if not current_regions:
                raise click.ClickException(f"❌ No regions available for role '{role}'. Check ai.yaml")
            
            # Get available deployment names for this role
            names = get_deployment_names(ai_config.data, current_regions, role, selected_platforms)
            
            if not names:
                logger.warning(f"⚠️  No {role} models found in regions {', '.join(sorted(current_regions))}")
                continue
            
            # Select deployment
            current_default = role_deployments[role] or os.getenv(role_deployment_env_var_name(role))
            
            if non_interactive:
                if current_default and current_default in names:
                    selected_deployment = current_default
                else:
                    selected_deployment = names[0]
                console.print(f"🎯 Selected [cyan]{role}[/cyan]: [green]{selected_deployment}[/green]")
            else:
                selected_deployment = select_model_interactive(role, names, current_default)
            
            # Get descriptor and update regions
            descriptor = ai_config.descriptors[selected_deployment]
            current_regions = current_regions & descriptor.regions
            
            # Track selected platform for teacher/student roles
            platform = descriptor.data.get('platform')
            if platform and role in ['teacher', 'student']:
                selected_platforms[role] = platform
            
            # Store selections
            role_selections = [
                (role_deployment_env_var_name(role), selected_deployment),
                (role_model_env_var_name(role), descriptor.model.name),
                (role_model_api_env_var_name(role), descriptor.model.api)
            ]
            
            # Check for finetuning SKU for this specific deployment
            finetuning_sku = get_finetuning_sku_name(ai_config.data, selected_deployment)
            if finetuning_sku:
                role_selections.append((role_finetuning_sku_env_var_name(role), finetuning_sku))
                logger.info(f"✅ Found finetuning SKU for {role}: {finetuning_sku}")
            
            selections.extend(role_selections)
            
            logger.info(f"✅ Selected {role}: {selected_deployment} ({descriptor.model.name})")
        
        # Select final region
        available_regions = list(current_regions)
        if not available_regions:
            raise click.ClickException("❌ No common regions available for all selected models")
        
        current_location = os.getenv("AZURE_LOCATION")
        if non_interactive:
            if current_location and current_location in available_regions:
                selected_region = current_location
            else:
                selected_region = available_regions[0]
            console.print(f"🌍 Selected region: [green]{selected_region}[/green]")
        else:
            selected_region = select_region_interactive(available_regions, current_location)
        
        selections.append(("AZURE_LOCATION", selected_region))
        
        # Display configuration summary
        display_configuration_summary(selections, selected_region)
        
        # Set environment variables
        if set_azd_env:
            console.print("\n💾 [bold]Saving configuration to azd environment:[/bold]")
            success_count = 0
            for name, value in selections:
                if azd_set_env(name, value):
                    console.print(f"   ✅ {name}={value}")
                    success_count += 1
                else:
                    console.print(f"   ❌ Failed to set {name}={value}")
            
            console.print(f"\n🎉 Successfully configured {success_count}/{len(selections)} settings!")
        else:
            console.print("\n📋 [dim]Configuration complete (environment variables not set)[/dim]")
        
        console.print("\n🎯 [bold]Next Steps:[/bold]")
        console.print("   • Your model configuration is now ready for RAFT workflows")
        console.print("   • Run 'raft gen' to generate synthetic datasets with the configured models")
        console.print("   • Use 'raft status' to view current configuration")
        console.print(f"   • Selected region: [cyan]{selected_region}[/cyan]")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if verbose:
            console.print_exception()
        raise click.ClickException(f"Configuration failed: {e}")
