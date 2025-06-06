#!/usr/bin/env python3
"""
Test runner for the Mirror Mirror system.
This script helps start, stop, and test the complete pipeline.
"""

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.live import Live

app = typer.Typer()
console = Console()


class ProcessManager:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        
    def start_component(self, name: str, command: List[str], cwd: Path = None) -> bool:
        """Start a component process"""
        try:
            if name in self.processes:
                console.print(f"[yellow]Component {name} already running[/yellow]")
                return True
                
            console.print(f"[blue]Starting {name}...[/blue]")
            process = subprocess.Popen(
                command,
                cwd=cwd or Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes[name] = process
            
            # Give it a moment to start
            time.sleep(1)
            
            if process.poll() is None:
                console.print(f"[green]✓ {name} started (PID: {process.pid})[/green]")
                return True
            else:
                stdout, stderr = process.communicate()
                console.print(f"[red]✗ {name} failed to start[/red]")
                console.print(f"[red]STDOUT: {stdout}[/red]")
                console.print(f"[red]STDERR: {stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]✗ Failed to start {name}: {e}[/red]")
            return False
    
    def stop_component(self, name: str) -> bool:
        """Stop a component process"""
        if name not in self.processes:
            return True
            
        process = self.processes[name]
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            del self.processes[name]
            console.print(f"[green]✓ {name} stopped[/green]")
            return True
        except Exception as e:
            console.print(f"[red]✗ Failed to stop {name}: {e}[/red]")
            return False
    
    def stop_all(self):
        """Stop all running processes"""
        for name in list(self.processes.keys()):
            self.stop_component(name)
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all components"""
        status = {}
        for name, process in self.processes.items():
            if process.poll() is None:
                status[name] = "Running"
            else:
                status[name] = "Stopped"
        return status


manager = ProcessManager()


@app.command()
def start_redis():
    """Start Redis using Docker Compose"""
    console.print("[blue]Starting Redis...[/blue]")
    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "redis"],
            capture_output=True,
            text=True,
            check=True
        )
        console.print("[green]✓ Redis started[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Failed to start Redis: {e.stderr}[/red]")
        return False


@app.command()
def stop_redis():
    """Stop Redis using Docker Compose"""
    console.print("[blue]Stopping Redis...[/blue]")
    try:
        subprocess.run(
            ["docker", "compose", "down"],
            capture_output=True,
            text=True,
            check=True
        )
        console.print("[green]✓ Redis stopped[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Failed to stop Redis: {e.stderr}[/red]")


@app.command()
def start_pipeline(
    mode: str = typer.Option("fake", help="Diffusion mode: fake or sdxs"),
    camera_id: int = typer.Option(0, help="Camera device ID"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Start the complete processing pipeline"""
    
    # Set logging level
    log_level = "DEBUG" if debug else "INFO"
    
    components = [
        ("camera", ["uv", "run", "python", "-m", "mirror_mirror.camera_server"], {"CAMERA_ID": str(camera_id)}),
        ("latent_encoder", ["uv", "run", "python", "-m", "mirror_mirror.latent_encoder"], {}),
        ("diffusion", ["uv", "run", "python", "-m", "mirror_mirror.diffusion_server"], {"MODE": mode}),
        ("latent_decoder", ["uv", "run", "python", "-m", "mirror_mirror.latent_decoder"], {}),
        ("display", ["uv", "run", "python", "-m", "mirror_mirror.display"], {}),
    ]
    
    src_path = Path("src")
    
    for name, command, env_vars in components:
        # Set environment variables
        env = {**env_vars, "PYTHONPATH": str(src_path), "LOG_LEVEL": log_level}
        
        success = manager.start_component(
            name, 
            command,
            cwd=Path.cwd()
        )
        
        if not success:
            console.print(f"[red]Failed to start {name}, stopping pipeline[/red]")
            manager.stop_all()
            return
        
        time.sleep(2)  # Give each component time to initialize
    
    console.print("[green]✓ Pipeline started successfully![/green]")
    console.print("[yellow]Press Ctrl+C to stop all components[/yellow]")
    
    try:
        # Monitor components
        while True:
            time.sleep(5)
            status = manager.get_status()
            
            # Check if any component died
            for name, state in status.items():
                if state == "Stopped":
                    console.print(f"[red]Component {name} stopped unexpectedly![/red]")
                    raise KeyboardInterrupt
                    
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping pipeline...[/yellow]")
        manager.stop_all()


@app.command()
def status():
    """Show status of all components"""
    table = Table(title="Mirror Mirror System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("PID", style="yellow")
    
    # Check Redis
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "redis"],
            capture_output=True,
            text=True
        )
        redis_status = "Running" if "Up" in result.stdout else "Stopped"
    except:
        redis_status = "Unknown"
    
    table.add_row("Redis", redis_status, "-")
    
    # Check pipeline components
    status = manager.get_status()
    for name, state in status.items():
        pid = str(manager.processes[name].pid) if name in manager.processes else "-"
        table.add_row(name, state, pid)
    
    console.print(table)


@app.command()
def test_simple():
    """Run a simple test with fake diffusion"""
    console.print("[blue]Running simple test with fake diffusion...[/blue]")
    
    # Start Redis
    if not start_redis():
        return
    
    time.sleep(3)
    
    # Start pipeline in fake mode
    try:
        start_pipeline(mode="fake", debug=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_redis()


@app.command() 
def test_full():
    """Run full test with real diffusion (requires GPU)"""
    console.print("[blue]Running full test with SDXS diffusion...[/blue]")
    
    # Start Redis
    if not start_redis():
        return
    
    time.sleep(3)
    
    # Start pipeline in SDXS mode
    try:
        start_pipeline(mode="sdxs", debug=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_redis()


@app.command()
def publish_prompt(prompt: str):
    """Publish a test prompt to the system"""
    console.print(f"[blue]Publishing prompt: '{prompt}'[/blue]")
    
    try:
        result = subprocess.run([
            "uv", "run", "python", "-m", "mirror_mirror.prompt_publisher", prompt
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✓ Prompt published[/green]")
        else:
            console.print(f"[red]✗ Failed to publish prompt: {result.stderr}[/red]")
            
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")


@app.command()
def cleanup():
    """Stop all components and cleanup"""
    console.print("[blue]Cleaning up system...[/blue]")
    manager.stop_all()
    stop_redis()
    console.print("[green]✓ Cleanup complete[/green]")


if __name__ == "__main__":
    # Setup rich logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted, cleaning up...[/yellow]")
        manager.stop_all()
        sys.exit(0) 