# PowerShell script to start the orchestration system
# Usage: .\orchestration\run_orchestration.ps1

param(
    [int]$Workers = 3,
    [switch]$DryRun,
    [switch]$Interactive,
    [switch]$GeneratePrompts
)

$ErrorActionPreference = "Stop"

# Check for API key
if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "ERROR: ANTHROPIC_API_KEY not set" -ForegroundColor Red
    Write-Host "Set it with: `$env:ANTHROPIC_API_KEY='your-key'" -ForegroundColor Yellow
    exit 1
}

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HEAT SIGNATURE ZERO - ORCHESTRATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"
Write-Host "Workers: $Workers"
Write-Host ""

# Create shared directory
$SharedDir = Join-Path $ScriptDir "shared"
if (-not (Test-Path $SharedDir)) {
    New-Item -ItemType Directory -Path $SharedDir | Out-Null
    Write-Host "Created shared directory: $SharedDir"
}

# Generate prompts
Write-Host "Generating worker prompts..." -ForegroundColor Yellow
Push-Location $ProjectDir
python -c "
import sys
sys.path.insert(0, '.')
from orchestration.worker_prompts import generate_all_prompts
from pathlib import Path

prompts = generate_all_prompts()
shared_dir = Path('orchestration/shared')
shared_dir.mkdir(exist_ok=True)

for worker_id, prompt in prompts.items():
    prompt_file = shared_dir / f'prompt_{worker_id}.txt'
    prompt_file.write_text(prompt)
    print(f'  Generated: {prompt_file}')
"
Pop-Location

if ($GeneratePrompts) {
    Write-Host ""
    Write-Host "Prompts generated. Use -Interactive or no flag to start workers." -ForegroundColor Green
    exit 0
}

if ($DryRun) {
    Write-Host ""
    Write-Host "[DRY RUN] Would start $Workers workers" -ForegroundColor Yellow
    Write-Host ""

    for ($i = 1; $i -le $Workers; $i++) {
        $PromptFile = Join-Path $SharedDir "prompt_W$i.txt"
        Write-Host "Worker W$i prompt:" -ForegroundColor Cyan
        Get-Content $PromptFile | Select-Object -First 20
        Write-Host "..." -ForegroundColor Gray
        Write-Host ""
    }
    exit 0
}

if ($Interactive) {
    Write-Host ""
    Write-Host "INTERACTIVE MODE" -ForegroundColor Green
    Write-Host "================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Starting $Workers terminal windows..." -ForegroundColor Yellow
    Write-Host ""

    for ($i = 1; $i -le $Workers; $i++) {
        $ContainerName = "claude-worker-$i"
        $WorkerId = "W$i"

        # Check if container exists, create if not
        $existing = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}" 2>$null

        if ($existing) {
            Write-Host "Using existing container: $ContainerName" -ForegroundColor Gray
        } else {
            Write-Host "Creating container: $ContainerName" -ForegroundColor Yellow
            docker run -d `
                --name $ContainerName `
                -v "${ProjectDir}:/workspace" `
                -e "ANTHROPIC_API_KEY=$env:ANTHROPIC_API_KEY" `
                -e "WORKER_ID=$WorkerId" `
                --cpus 8 `
                --memory 32g `
                -w /workspace `
                heat-signature-zero:latest `
                tail -f /dev/null
        }

        # Start new terminal with docker exec
        $cmd = "docker exec -it $ContainerName bash -c 'echo Worker $WorkerId && cat /workspace/orchestration/shared/prompt_$WorkerId.txt && echo && echo Run: claude --dangerously-skip-permissions && bash'"
        Start-Process "cmd.exe" -ArgumentList "/k", $cmd

        Write-Host "Started terminal for $WorkerId" -ForegroundColor Green
        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "All terminals started!" -ForegroundColor Green
    Write-Host "In each terminal:" -ForegroundColor Yellow
    Write-Host "  1. Run: claude --dangerously-skip-permissions" -ForegroundColor White
    Write-Host "  2. Paste the prompt shown" -ForegroundColor White
    Write-Host ""

} else {
    # Non-interactive: run orchestrator Python script
    Write-Host ""
    Write-Host "Starting Python orchestrator..." -ForegroundColor Yellow
    Push-Location $ProjectDir
    python orchestration/orchestrator.py --workers $Workers
    Pop-Location
}
