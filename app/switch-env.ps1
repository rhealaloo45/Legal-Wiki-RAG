# Copies the per-branch env file over app/.env so DATABASE_URL matches
# whichever branch you're about to work on. Run by hand after `git checkout`
# — nothing here is wired into git, on purpose (see README "Switching
# branches with multiple databases").
#
# Usage:
#   .\switch-env.ps1 -Branch optimize   # app/.env -> legal_wiki
#   .\switch-env.ps1 -Branch v2         # app/.env -> legalwiki_v2

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("optimize", "v2")]
    [string]$Branch
)

$source = Join-Path $PSScriptRoot ".env.$Branch"
$target = Join-Path $PSScriptRoot ".env"

if (-not (Test-Path $source)) {
    Write-Error "Missing $source"
    exit 1
}

Copy-Item $source $target -Force
Write-Host "app/.env now points at the '$Branch' branch's DB (copied from .env.$Branch)"
