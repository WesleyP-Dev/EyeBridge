$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    $localPython = Get-ChildItem -LiteralPath '.python' -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -like 'cpython-3.1[12]*' } | Select-Object -First 1
    if ($localPython) {
        & $localPython.FullName -m venv .venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $compatible = $false
        foreach ($version in @('-3.11', '-3.12')) {
            & py $version -c 'import sys; print(sys.version)' 2>$null
            if ($LASTEXITCODE -eq 0) {
                & py $version -m venv .venv
                $compatible = $LASTEXITCODE -eq 0
                break
            }
        }
        if (-not $compatible) {
            throw 'Instale Python 3.11 ou 3.12 (64-bit) de https://www.python.org/downloads/windows/ e execute novamente.'
        }
    } else {
        throw 'Instale Python 3.11 ou 3.12 (64-bit) de https://www.python.org/downloads/windows/ e execute novamente.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel criar o ambiente virtual.' }
}
& .\.venv\Scripts\python.exe -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Use Python 3.11/3.12"'
if ($LASTEXITCODE -ne 0) { throw 'Ambiente Python incompativel.' }
& .\.venv\Scripts\python.exe -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel preparar pip.' }
& .\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-windows.lock
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar dependencias.' }
& .\.venv\Scripts\python.exe main.py --check
if ($LASTEXITCODE -ne 0) { throw 'Falha ao validar modelo.' }
Write-Host 'Pronto. Abra iniciar_windows.cmd ou demonstracao_windows.cmd.'
