# Measure how a llama.cpp server scales with concurrent slots.
#
# Answers "how many people can this card serve": for each --parallel value it
# records the single-request decode speed, the aggregate throughput of that many
# simultaneous requests, and the speed each of those requests sees.
#
# The server must already be built (see docs/benchmarks/concurrency.md); this
# script only drives it. The GPU has to be free: stop any other daemon first.
#
#   .\scripts\bench-concurrency.ps1 -ServerExe C:\path\llama-server.exe `
#       -Model C:\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf `
#       -Parallel 1,2,4,8 -Spec 'none','mtp'
#
# Writes one JSON line per configuration to -Out (default: bench-concurrency.json).

param(
    [Parameter(Mandatory = $true)][string]$ServerExe,
    [Parameter(Mandatory = $true)][string]$Model,
    [int[]]$Parallel = @(1, 2, 4, 8),
    # 'none' and/or 'mtp'. MTP needs a checkpoint that ships a head.
    [string[]]$Spec = @('none', 'mtp'),
    [int]$Ctx = 32768,
    [int]$MaxTokens = 128,
    [int]$Port = 15810,
    [string]$Out = 'bench-concurrency.json'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
$base = "http://127.0.0.1:$Port"

function Wait-Health([int]$seconds) {
    for ($i = 0; $i -lt $seconds; $i++) {
        try { Invoke-RestMethod "$base/health" -TimeoutSec 3 | Out-Null; return $true } catch { Start-Sleep -Seconds 1 }
    }
    return $false
}

function Chat([string]$prompt, [int]$tokens) {
    $req = @{
        messages = @(@{ role = 'user'; content = $prompt })
        max_tokens = $tokens; stream = $false; temperature = 0.8
    } | ConvertTo-Json -Depth 4
    $c = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType 'application/json' `
        -Body $req -TimeoutSec 900
    return $c.timings
}

# N requests at once, each its own conversation so no slot cache is shared and
# the measurement is not helped by a prompt the server has already seen.
function Load-Test([int]$n) {
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromMinutes(20)
    $tasks = @()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    for ($i = 1; $i -le $n; $i++) {
        $body = @{
            messages = @(@{ role = 'user'; content = "Session $i. Write one short paragraph about topic number $i, then stop." })
            max_tokens = $MaxTokens; stream = $false; temperature = 0.8
        } | ConvertTo-Json -Depth 4
        $msg = New-Object System.Net.Http.HttpRequestMessage ([System.Net.Http.HttpMethod]::Post), "$base/v1/chat/completions"
        $msg.Content = New-Object System.Net.Http.StringContent ($body, [System.Text.Encoding]::UTF8, 'application/json')
        $tasks += $client.SendAsync($msg)
    }
    [System.Threading.Tasks.Task]::WaitAll($tasks)
    $sw.Stop()

    $perRequest = @()
    $tokens = 0
    foreach ($t in $tasks) {
        $json = $t.Result.Content.ReadAsStringAsync().Result | ConvertFrom-Json
        if ($json.timings) {
            $perRequest += [double]$json.timings.predicted_per_second
            $tokens += [int]$json.timings.predicted_n
        }
    }
    $client.Dispose()
    $sorted = @($perRequest | Sort-Object)
    [pscustomobject]@{
        tokens           = $tokens
        wall_seconds     = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        aggregate_tps    = if ($sw.Elapsed.TotalSeconds -gt 0) { [math]::Round($tokens / $sw.Elapsed.TotalSeconds, 1) } else { 0 }
        per_request_med  = if ($sorted.Count -gt 0) { [math]::Round($sorted[[int][math]::Floor($sorted.Count / 2)], 1) } else { 0 }
    }
}

$results = @()
foreach ($variant in $Spec) {
    foreach ($np in $Parallel) {
        $srvArgs = @('-m', $Model, '-ngl', '999', '-c', "$Ctx", '-np', "$np", '-fa', 'on',
                     '-ctk', 'f16', '-ctv', 'f16', '-b', '2048', '-ub', '512', '-cb',
                     '--host', '127.0.0.1', '--port', "$Port")
        if ($variant -eq 'mtp') { $srvArgs += @('--spec-type', 'draft-mtp') }

        $log = Join-Path $env:TEMP "bench-$variant-$np.log"
        $proc = Start-Process -FilePath $ServerExe -ArgumentList $srvArgs -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        try {
            if (-not (Wait-Health 300)) {
                Write-Warning "$variant np=$np : server did not come up (see $log.err)"
                continue
            }
            $null = Chat 'Say hello.' 16                    # warm up
            $single = Chat 'Write one short paragraph about the sea, then stop.' $MaxTokens
            $load = Load-Test $np
            $row = [pscustomobject]@{
                spec            = $variant
                parallel        = $np
                single_tps      = [math]::Round([double]$single.predicted_per_second, 1)
                aggregate_tps   = $load.aggregate_tps
                per_request_med = $load.per_request_med
                wall_seconds    = $load.wall_seconds
                vram_mib        = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            }
            $results += $row
            $row | Format-Table -AutoSize | Out-String | Write-Host
        } finally {
            if (-not $proc.HasExited) { $proc | Stop-Process -Force }
            Start-Sleep -Seconds 4
        }
    }
}

$results | ConvertTo-Json -Depth 4 | Set-Content -Path $Out -Encoding UTF8
"wrote $Out"
$results | Format-Table -AutoSize
