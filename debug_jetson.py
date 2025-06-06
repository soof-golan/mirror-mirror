#!/usr/bin/env python3
"""
Debug and monitoring tool for Mirror Mirror on Jetson Orin NX
"""

import asyncio
import json
import psutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List

import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer()
console = Console()


def get_jetson_stats() -> Dict[str, Any]:
    """Get Jetson-specific system stats"""
    stats = {}
    
    # Try to get Jetson stats from tegrastats
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "100", "--stop"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Parse tegrastats output
            stats['tegrastats'] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        stats['tegrastats'] = "Not available"
    
    # GPU memory usage
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            mem_used, mem_total, temp, util = result.stdout.strip().split(", ")
            stats['gpu'] = {
                'memory_used_mb': int(mem_used),
                'memory_total_mb': int(mem_total),
                'temperature_c': int(temp),
                'utilization_pct': int(util)
            }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        stats['gpu'] = None
    
    return stats


def get_system_stats() -> Dict[str, Any]:
    """Get general system stats"""
    stats = {}
    
    # CPU
    stats['cpu'] = {
        'percent': psutil.cpu_percent(interval=1),
        'count': psutil.cpu_count(),
        'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    }
    
    # Memory
    mem = psutil.virtual_memory()
    stats['memory'] = {
        'total_gb': mem.total / 1024**3,
        'used_gb': mem.used / 1024**3,
        'percent': mem.percent
    }
    
    # Disk
    disk = psutil.disk_usage('/')
    stats['disk'] = {
        'total_gb': disk.total / 1024**3,
        'used_gb': disk.used / 1024**3,
        'percent': (disk.used / disk.total) * 100
    }
    
    # Processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            if 'mirror_mirror' in proc.info['name'] or 'python' in proc.info['name']:
                processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    stats['processes'] = processes[:10]  # Top 10
    
    return stats


def check_camera_devices() -> List[Dict[str, Any]]:
    """Check available camera devices"""
    devices = []
    
    # Check /dev/video* devices
    video_devices = list(Path("/dev").glob("video*"))
    
    for device in video_devices:
        device_info = {'path': str(device)}
        
        try:
            # Try to get device capabilities
            result = subprocess.run(
                ["v4l2-ctl", "--device", str(device), "--list-formats-ext"],
                capture_output=True,
                text=True,
                timeout=5
            )
            device_info['v4l2_info'] = result.stdout if result.returncode == 0 else "Error"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            device_info['v4l2_info'] = "v4l2-ctl not available"
        
        # Test with OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(str(device))
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                device_info['opencv_test'] = f"{int(width)}x{int(height)} @ {fps:.1f}fps"
                cap.release()
            else:
                device_info['opencv_test'] = "Failed to open"
        except Exception as e:
            device_info['opencv_test'] = f"Error: {e}"
        
        devices.append(device_info)
    
    return devices


def check_redis_connection() -> Dict[str, Any]:
    """Check Redis connection"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        info = r.info()
        return {
            'connected': True,
            'version': info.get('redis_version', 'unknown'),
            'memory_used': info.get('used_memory_human', 'unknown'),
            'clients': info.get('connected_clients', 0)
        }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e)
        }


@app.command()
def monitor():
    """Real-time system monitoring"""
    console.print("[blue]Starting real-time monitoring...[/blue]")
    console.print("Press Ctrl+C to stop")
    
    def create_layout():
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        return layout
    
    def update_display():
        layout = create_layout()
        
        # Header
        layout["header"].update(Panel("🪞 Mirror Mirror - Jetson Monitor", style="bold blue"))
        
        # System stats
        sys_stats = get_system_stats()
        jetson_stats = get_jetson_stats()
        
        # Left panel - System info
        system_table = Table(title="System Resources")
        system_table.add_column("Resource", style="cyan")
        system_table.add_column("Usage", style="green")
        
        system_table.add_row("CPU", f"{sys_stats['cpu']['percent']:.1f}%")
        system_table.add_row("Memory", f"{sys_stats['memory']['percent']:.1f}% ({sys_stats['memory']['used_gb']:.1f}GB)")
        system_table.add_row("Disk", f"{sys_stats['disk']['percent']:.1f}% ({sys_stats['disk']['used_gb']:.1f}GB)")
        
        if jetson_stats.get('gpu'):
            gpu = jetson_stats['gpu']
            system_table.add_row("GPU Memory", f"{gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB")
            system_table.add_row("GPU Temp", f"{gpu['temperature_c']}°C")
            system_table.add_row("GPU Util", f"{gpu['utilization_pct']}%")
        
        layout["left"].update(system_table)
        
        # Right panel - Processes
        proc_table = Table(title="Mirror Mirror Processes")
        proc_table.add_column("PID", style="yellow")
        proc_table.add_column("Name", style="cyan")
        proc_table.add_column("CPU%", style="green")
        proc_table.add_column("Mem%", style="blue")
        
        for proc in sys_stats['processes']:
            proc_table.add_row(
                str(proc['pid']),
                proc['name'][:20],
                f"{proc['cpu_percent']:.1f}",
                f"{proc['memory_percent']:.1f}"
            )
        
        layout["right"].update(proc_table)
        
        # Footer
        redis_info = check_redis_connection()
        redis_status = "🟢 Connected" if redis_info['connected'] else "🔴 Disconnected"
        layout["footer"].update(Panel(f"Redis: {redis_status} | Time: {time.strftime('%H:%M:%S')}", style="dim"))
        
        return layout
    
    try:
        with Live(update_display(), refresh_per_second=2, screen=True):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


@app.command()
def diagnose():
    """Run comprehensive system diagnostics"""
    console.print("[blue]Running diagnostics...[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # System check
        task1 = progress.add_task("Checking system resources...", total=None)
        sys_stats = get_system_stats()
        jetson_stats = get_jetson_stats()
        progress.update(task1, completed=True)
        
        # Camera check
        task2 = progress.add_task("Testing camera devices...", total=None)
        cameras = check_camera_devices()
        progress.update(task2, completed=True)
        
        # Redis check
        task3 = progress.add_task("Testing Redis connection...", total=None)
        redis_info = check_redis_connection()
        progress.update(task3, completed=True)
        
        # Python environment check
        task4 = progress.add_task("Checking Python environment...", total=None)
        env_check = check_python_environment()
        progress.update(task4, completed=True)
    
    # Display results
    console.print("\n[bold green]Diagnostics Results[/bold green]")
    console.print("=" * 50)
    
    # System
    console.print(f"\n[bold]System Resources:[/bold]")
    console.print(f"  CPU: {sys_stats['cpu']['percent']:.1f}%")
    console.print(f"  Memory: {sys_stats['memory']['percent']:.1f}% ({sys_stats['memory']['used_gb']:.1f}GB)")
    
    if jetson_stats.get('gpu'):
        gpu = jetson_stats['gpu']
        console.print(f"  GPU Memory: {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB")
        console.print(f"  GPU Temperature: {gpu['temperature_c']}°C")
    
    # Cameras
    console.print(f"\n[bold]Camera Devices:[/bold]")
    for i, cam in enumerate(cameras):
        status = "✅" if "Failed" not in cam['opencv_test'] else "❌"
        console.print(f"  {status} {cam['path']}: {cam['opencv_test']}")
    
    # Redis
    console.print(f"\n[bold]Redis:[/bold]")
    if redis_info['connected']:
        console.print(f"  ✅ Connected (v{redis_info['version']})")
    else:
        console.print(f"  ❌ Connection failed: {redis_info['error']}")
    
    # Python environment
    console.print(f"\n[bold]Python Environment:[/bold]")
    for lib, status in env_check.items():
        status_icon = "✅" if status['available'] else "❌"
        console.print(f"  {status_icon} {lib}: {status['version'] if status['available'] else status['error']}")


def check_python_environment() -> Dict[str, Dict[str, Any]]:
    """Check Python library availability"""
    libraries = ['cv2', 'torch', 'diffusers', 'faststream', 'redis', 'PIL']
    results = {}
    
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'unknown')
            results[lib] = {'available': True, 'version': version}
        except ImportError as e:
            results[lib] = {'available': False, 'error': str(e)}
    
    # Special checks
    try:
        import torch
        results['torch']['cuda'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results['torch']['device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    return results


@app.command()
def test_components():
    """Test individual components quickly"""
    console.print("[blue]Testing individual components...[/blue]")
    
    tests = [
        ("Camera Test", test_camera_quick),
        ("Redis Test", test_redis_quick),
        ("PyTorch/CUDA Test", test_pytorch_quick),
        ("Diffusers Test", test_diffusers_quick),
    ]
    
    for name, test_func in tests:
        console.print(f"\n[yellow]Running {name}...[/yellow]")
        try:
            result = test_func()
            if result:
                console.print(f"  ✅ {name} passed")
            else:
                console.print(f"  ❌ {name} failed")
        except Exception as e:
            console.print(f"  ❌ {name} error: {e}")


def test_camera_quick() -> bool:
    """Quick camera test"""
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    return False


def test_redis_quick() -> bool:
    """Quick Redis test"""
    import redis
    r = redis.Redis(host='localhost', port=6379)
    try:
        r.ping()
        return True
    except:
        return False


def test_pytorch_quick() -> bool:
    """Quick PyTorch/CUDA test"""
    import torch
    x = torch.randn(2, 2)
    if torch.cuda.is_available():
        x = x.cuda()
        y = x + 1
        return y.is_cuda
    return True


def test_diffusers_quick() -> bool:
    """Quick diffusers test"""
    try:
        from diffusers import StableDiffusionPipeline
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    app() 