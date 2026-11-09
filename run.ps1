# One command to install and open the demo (Windows, in PowerShell):
#   irm https://raw.githubusercontent.com/Meditect/yako-anemia-eye/main/run.ps1 | iex
# Everything goes into one folder ($HOME\yako-anemia-eye) and a private Python environment;
# nothing is installed system-wide. Run it again to restart the demo.
$ErrorActionPreference = 'Stop'
$Repo = 'https://github.com/Meditect/yako-anemia-eye'
$Dir = if ($env:YAKO_DIR) { $env:YAKO_DIR } else { Join-Path $HOME 'yako-anemia-eye' }
$Port = if ($env:PORT) { $env:PORT } else { 8000 }
function Say($m) { Write-Host "`n==> $m" -ForegroundColor Green }
function Die($m) { Write-Host "`nError: $m" -ForegroundColor Red; exit 1 }

# 1. Python 3.10 or later (the "py" launcher comes with the python.org installer)
function Test-Python($exe, [string[]]$pre) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { return $false }
    try { & $exe @pre -c "import sys; sys.exit(sys.version_info < (3, 10))" *> $null } catch { return $false }
    return $LASTEXITCODE -eq 0
}
$Py = $null; $PyArgs = @()
if (Test-Python 'py' @('-3')) { $Py = 'py'; $PyArgs = @('-3') }
elseif (Test-Python 'python' @()) { $Py = 'python' }
if (-not $Py) { Die 'Python 3.10 or later is needed. Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH") and run this command again.' }

# 2. The code
if (-not (Test-Path (Join-Path $Dir 'anemia_eye\pipeline.py'))) {
    Say "Downloading the code into $Dir"
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git clone -q "$Repo.git" $Dir
    } else {
        $zip = Join-Path $env:TEMP 'yako-anemia-eye.zip'
        Invoke-WebRequest "$Repo/archive/refs/heads/main.zip" -OutFile $zip
        Expand-Archive $zip -DestinationPath $env:TEMP -Force
        Move-Item (Join-Path $env:TEMP 'yako-anemia-eye-main') $Dir
    }
} elseif ((Test-Path (Join-Path $Dir '.git')) -and (Get-Command git -ErrorAction SilentlyContinue)) {
    Say "Updating the code in $Dir"
    git -C $Dir pull -q --ff-only
}
Set-Location $Dir

# 3. Private Python environment and libraries (first time only, a few minutes)
$VPy = Join-Path $Dir '.venv\Scripts\python.exe'
if (-not (Test-Path $VPy)) {
    Say 'Creating a private Python environment'
    & $Py @PyArgs -m venv .venv
}
$Marker = Join-Path $Dir '.venv\.installed'
if (-not (Test-Path $Marker) -or ((Get-Content requirements.txt -Raw) -ne (Get-Content $Marker -Raw))) {
    Say 'Installing the libraries (first time only, a few minutes)'
    & $VPy -m pip install -q --upgrade pip
    & $VPy -m pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Die 'The installation of the libraries failed (see above).' }
    Copy-Item requirements.txt $Marker
}

# 4. Model weights (125 MB, first time only)
& $VPy -c "from anemia_eye import download_weights; download_weights()"

# 5. Start the demo and open the browser once it answers
$Url = "http://localhost:$Port"
Say "Starting the demo on $Url (press Ctrl+C to stop)"
Start-Job -ArgumentList $Url -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 120; $i++) {
        try { Invoke-WebRequest "$u/api/health" -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; break } catch { Start-Sleep 1 }
    }
} | Out-Null
& $VPy -m uvicorn server:app --host 127.0.0.1 --port $Port
