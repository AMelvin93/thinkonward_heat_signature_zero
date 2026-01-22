# Worker loop wrapper - restarts Claude Code after each experiment
# This implements the "Ralph Wiggum loop" with context clearing
#
# Usage:
#   .\orchestration\run_worker_loop.ps1 -WorkerId W1
#   .\orchestration\run_worker_loop.ps1 -WorkerId W2
#   etc.

param(
    [string]$WorkerId = "W1"
)

$PromptFile = "orchestration\shared\prompt_$WorkerId.txt"
$StopFile = "orchestration\shared\STOP"

Write-Host "=============================================="
Write-Host "Worker Loop: $WorkerId (Context-Clearing Mode)"
Write-Host "=============================================="
Write-Host "Each experiment runs in a fresh Claude Code instance."
Write-Host "Context is cleared between experiments."
Write-Host "Create $StopFile to stop the loop."
Write-Host "=============================================="

# Main loop - runs forever until STOP file created
while ($true) {
    # Check for stop file
    if (Test-Path $StopFile) {
        Write-Host "[$WorkerId] STOP file detected. Exiting loop."
        exit 0
    }

    # Check if prompt file exists
    if (-not (Test-Path $PromptFile)) {
        Write-Host "[$WorkerId] Prompt file not found: $PromptFile"
        Write-Host "[$WorkerId] Run: uv run python orchestration/worker_prompts_v4.py"
        exit 1
    }

    Write-Host ""
    Write-Host "[$WorkerId] Starting Claude Code instance... ($(Get-Date))"
    Write-Host "[$WorkerId] Context will be cleared after this experiment."
    Write-Host ""

    # Read prompt
    $Prompt = Get-Content -Path $PromptFile -Raw

    # Run Claude Code with the prompt
    # --dangerously-skip-permissions: Auto-approve all tool calls
    # --print: Print responses (not interactive mode)
    # The worker will exit after completing one experiment
    $process = Start-Process -FilePath "claude" -ArgumentList "--dangerously-skip-permissions", "--print", "`"$Prompt`"" -Wait -PassThru -NoNewWindow

    $exitCode = $process.ExitCode

    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "[$WorkerId] Claude Code exited cleanly (experiment completed)"
        Write-Host "[$WorkerId] Restarting with fresh context in 5 seconds..."
        Start-Sleep -Seconds 5
    } else {
        Write-Host ""
        Write-Host "[$WorkerId] Claude Code exited with code $exitCode"
        Write-Host "[$WorkerId] Waiting 30 seconds before retry..."
        Start-Sleep -Seconds 30
    }
}
