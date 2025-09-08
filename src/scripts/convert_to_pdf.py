#!/usr/bin/env python3
"""
Convert markdown articles to PDF using markdown2pdf command.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional

import click
import rich_click as rich_click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import logging

# Configure rich click
rich_click.rich_click.USE_RICH_MARKUP = True
rich_click.rich_click.USE_MARKDOWN = True

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(), rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)
console = Console()


def find_markdown_files(directory: Path, exclude_dirs: Optional[List[str]] = None) -> List[Path]:
    """Find all markdown files in the given directory, excluding specified subdirectories."""
    if exclude_dirs is None:
        exclude_dirs = []
    
    markdown_files = []
    
    for root, dirs, files in os.walk(directory):
        # Remove excluded directories from dirs list to prevent walking into them
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        root_path = Path(root)
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(root_path / file)
    
    return sorted(markdown_files)


def convert_markdown_to_pdf(markdown_file: Path, output_dir: Optional[Path] = None) -> bool:
    """Convert a single markdown file to PDF using markdown2pdf."""
    try:
        # Determine output path
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf_file = output_dir / f"{markdown_file.stem}.pdf"
        else:
            pdf_file = markdown_file.parent / f"{markdown_file.stem}.pdf"
        
        # Run markdown2pdf command with correct options
        cmd = [
            "markdown2pdf",
            "--path", str(markdown_file),
            "--output", str(pdf_file)
        ]
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ Converted: {markdown_file.name} -> {pdf_file.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to convert {markdown_file.name}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ markdown2pdf command not found. Please install it first.")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error converting {markdown_file.name}: {str(e)}")
        return False


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("articles"),
    help="Input directory containing markdown files",
    show_default=True
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory for PDF files (default: same as markdown file location)"
)
@click.option(
    "--exclude-dirs",
    "-e",
    multiple=True,
    default=["images", "org-chart"],
    help="Directories to exclude from processing",
    show_default=True
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Show what would be converted without actually converting"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@rich_click.rich_config(help_config=rich_click.RichHelpConfiguration(
    show_arguments=True,
    show_metavars_column=False,
    append_metavars_help=True,
))
def main(
    input_dir: Path,
    output_dir: Optional[Path],
    exclude_dirs: tuple,
    dry_run: bool,
    verbose: bool
):
    """
    Convert markdown articles to PDF using markdown2pdf command.
    
    This script will recursively find all .md files in the input directory
    and convert them to PDF format using the markdown2pdf command.
    
    **Examples:**
    
    ```bash
    # Convert all articles in the default articles directory
    uv run scripts/convert_to_pdf.py
    
    # Convert with custom input and output directories
    uv run scripts/convert_to_pdf.py -i ./my-articles -o ./pdfs
    
    # Dry run to see what would be converted
    uv run scripts/convert_to_pdf.py --dry-run
    
    # Exclude additional directories
    uv run scripts/convert_to_pdf.py -e images -e templates -e drafts
    ```
    """
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(f"\n🔍 [bold blue]Markdown to PDF Converter[/bold blue]")
    console.print(f"Input directory: [green]{input_dir.resolve()}[/green]")
    
    if output_dir:
        console.print(f"Output directory: [green]{output_dir.resolve()}[/green]")
    else:
        console.print("Output: [yellow]Same directory as source files[/yellow]")
    
    console.print(f"Excluding directories: [red]{', '.join(exclude_dirs)}[/red]")
    
    if dry_run:
        console.print("\n[yellow]🔍 DRY RUN MODE - No files will be converted[/yellow]")
    
    # Find markdown files
    logger.info(f"\nScanning for markdown files in {input_dir}...")
    markdown_files = find_markdown_files(input_dir, list(exclude_dirs))
    
    if not markdown_files:
        console.print("\n[red]❌ No markdown files found![/red]")
        return
    
    # Display summary table
    table = Table(title=f"\nFound {len(markdown_files)} Markdown Files")
    table.add_column("File", style="cyan")
    table.add_column("Directory", style="green")
    table.add_column("Size", justify="right", style="yellow")
    
    for md_file in markdown_files:
        relative_path = md_file.relative_to(input_dir)
        file_size = md_file.stat().st_size
        size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
        
        table.add_row(
            md_file.name,
            str(relative_path.parent) if relative_path.parent != Path(".") else ".",
            size_str
        )
    
    console.print(table)
    
    if dry_run:
        console.print("\n[yellow]✨ Dry run complete - no files were converted[/yellow]")
        return
    
    # Convert files
    console.print("\n[bold green]Starting conversion...[/bold green]")
    
    successful_conversions = 0
    failed_conversions = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        
        for md_file in markdown_files:
            task = progress.add_task(f"Converting {md_file.name}...", total=None)
            
            success = convert_markdown_to_pdf(md_file, output_dir)
            
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
            
            progress.remove_task(task)
    
    # Final summary
    console.print(f"\n[bold green]✨ Conversion Summary[/bold green]")
    summary_table = Table()
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    
    summary_table.add_row("✅ Successful", f"[green]{successful_conversions}[/green]")
    summary_table.add_row("❌ Failed", f"[red]{failed_conversions}[/red]")
    summary_table.add_row("📊 Total", f"[blue]{len(markdown_files)}[/blue]")
    
    console.print(summary_table)
    
    if failed_conversions > 0:
        console.print(f"\n[red]⚠️  {failed_conversions} conversions failed. Check the logs above for details.[/red]")
    else:
        console.print(f"\n[green]🎉 All conversions completed successfully![/green]")


if __name__ == "__main__":
    main()
