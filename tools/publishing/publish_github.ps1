# Requires Git, GitHub CLI and Python 3.11. Does not install software.
# Publishes the verified v1.0.0 package to KuznetsovKarazin/aura-ksp.
[CmdletBinding()]
param(
    [string]$ReleaseRoot = 'E:\AURA\AURA-KSP_PUBLIC_RELEASE_v1.0.0',
    [switch]$DraftOnly,
    [switch]$PrepareOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoName = 'KuznetsovKarazin/aura-ksp'
$ExpectedOwner = 'KuznetsovKarazin'
$RepoURL = "https://github.com/$RepoName"
$Tag = 'v1.0.0'
$RepoDir = Join-Path $ReleaseRoot 'repository'
$OutputDir = Join-Path $ReleaseRoot 'publish-files'
$Helper = Join-Path $PSScriptRoot 'prepare_publish.py'
$Notes = Join-Path $PSScriptRoot 'RELEASE_NOTES.md'

function Run-Tool {
    param([string]$Tool, [string[]]$ToolArgs)
    & $Tool @ToolArgs
    if ($LASTEXITCODE -ne 0) { throw "$Tool failed (exit $LASTEXITCODE). Publication stopped." }
}
function Probe-Tool {
    param([string]$Tool, [string[]]$ToolArgs)
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $Lines = @(& $Tool @ToolArgs 2>&1)
        $Code = $LASTEXITCODE
        return [PSCustomObject]@{ Code = $Code; Output = (($Lines | ForEach-Object { $_.ToString() }) -join "`n") }
    } finally { $ErrorActionPreference = $PreviousPreference }
}
function Require-Probe {
    param($Result, [string]$Description)
    if ($Result.Code -ne 0) { throw "$Description failed. $($Result.Output)" }
    return $Result.Output
}

foreach ($Command in @('git', 'py')) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) { throw "$Command is not installed or not in PATH. See START_HERE_RU.md." }
}
if (-not (Test-Path -LiteralPath $Helper)) { throw 'prepare_publish.py must be next to this script.' }
if (-not (Test-Path -LiteralPath $RepoDir)) { throw "Missing repository directory: $RepoDir" }
if (-not (Test-Path -LiteralPath (Join-Path $RepoDir 'LICENSE'))) { throw 'Final LICENSE is missing.' }
Run-Tool 'py' @('-3.11', $Helper, '--release-root', $ReleaseRoot, '--output-dir', $OutputDir)
if ($PrepareOnly) {
    Write-Host "LOCAL_FILES_READY: $OutputDir"
    Write-Host 'No remote operation was performed.'
    exit 0
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'GitHub CLI is not installed. Run: winget install --id GitHub.cli --exact --source winget; then reopen PowerShell.' }

$Auth = Probe-Tool 'gh' @('api', 'user')
if ($Auth.Code -ne 0) {
    Run-Tool 'gh' @('auth', 'login', '--hostname', 'github.com', '--git-protocol', 'https', '--web', '--scopes', 'workflow')
    $Auth = Probe-Tool 'gh' @('api', 'user')
}
$UserInfo = (Require-Probe $Auth 'GitHub authentication') | ConvertFrom-Json
if ($UserInfo.login -cne $ExpectedOwner) { throw "Authenticated as $($UserInfo.login); expected $ExpectedOwner. Run gh auth switch --user $ExpectedOwner and retry." }
$ScopeProbe = Probe-Tool 'gh' @('api', '--include', 'user')
if ($ScopeProbe.Code -eq 0) {
    $ScopeHeader = [regex]::Match($ScopeProbe.Output, '(?im)^x-oauth-scopes:\s*([^\r\n]*)')
    if ($ScopeHeader.Success -and 'workflow' -notin @($ScopeHeader.Groups[1].Value.Split(',') | ForEach-Object { $_.Trim() })) {
        throw 'The GitHub credential lacks workflow scope, required for .github/workflows. Run: gh auth refresh --hostname github.com --scopes workflow; then retry. No scope restriction was bypassed.'
    }
}
Run-Tool 'gh' @('auth', 'setup-git')

$RepositoryProbe = Probe-Tool 'gh' @('api', "repos/$RepoName")
$RemoteExists = $RepositoryProbe.Code -eq 0
if (-not $RemoteExists -and $RepositoryProbe.Output -notmatch 'HTTP 404') { throw "Cannot establish repository status. $($RepositoryProbe.Output)" }
if ($RemoteExists) {
    $RemoteInfo = $RepositoryProbe.Output | ConvertFrom-Json
    if ($RemoteInfo.full_name -ine $RepoName) { throw 'Unexpected remote repository identity.' }
    if ($RemoteInfo.private) { throw 'The existing repository is private; its visibility was not changed.' }
    $RemoteHeads = Require-Probe (Probe-Tool 'git' @('ls-remote', '--heads', "$RepoURL.git")) 'Remote branch inspection'
    if ($RemoteHeads.Trim() -and -not (Test-Path -LiteralPath (Join-Path $RepoDir '.git'))) {
        throw 'GitHub already contains commits, but this folder is not its clone. Nothing was overwritten. Use the existing clone or ask for the normal update procedure.'
    }
}

Push-Location $RepoDir
try {
    if (-not (Test-Path -LiteralPath '.git')) { Run-Tool 'git' @('init', '-b', 'main') }
    $Top = (Require-Probe (Probe-Tool 'git' @('rev-parse', '--show-toplevel')) 'Repository-root check').Trim()
    if ([IO.Path]::GetFullPath($Top).TrimEnd('\','/') -ine [IO.Path]::GetFullPath($RepoDir).TrimEnd('\','/')) { throw 'Unexpected Git working-tree root.' }
    $Branch = (Require-Probe (Probe-Tool 'git' @('branch', '--show-current')) 'Branch check').Trim()
    if ($Branch -ne 'main') { throw "Expected main branch; found $Branch. No branch was replaced." }
    Run-Tool 'git' @('config', 'core.autocrlf', 'false')
    $GitName = Probe-Tool 'git' @('config', 'user.name')
    if ($GitName.Code -ne 0 -or -not $GitName.Output.Trim()) { Run-Tool 'git' @('config', 'user.name', 'Oleksandr Kuznetsov') }
    $GitEmail = Probe-Tool 'git' @('config', 'user.email')
    if ($GitEmail.Code -ne 0 -or -not $GitEmail.Output.Trim()) {
        Run-Tool 'git' @('config', 'user.email', "$($UserInfo.id)+$ExpectedOwner@users.noreply.github.com")
    }
    Run-Tool 'git' @('add', '-A')
    Run-Tool 'py' @('-3.11', $Helper, '--release-root', $ReleaseRoot, '--check-index')
    $Changes = Probe-Tool 'git' @('diff', '--cached', '--quiet')
    if ($Changes.Code -eq 1) { Run-Tool 'git' @('commit', '-m', 'Release AURA-KSP v1.0.0 research artifact') }
    elseif ($Changes.Code -ne 0) { throw 'Cannot inspect staged changes.' }
    $Commit = (Require-Probe (Probe-Tool 'git' @('rev-parse', 'HEAD')) 'Commit check').Trim()

    $Origin = Probe-Tool 'git' @('remote', 'get-url', 'origin')
    if ($Origin.Code -eq 0) {
        $OriginURL = $Origin.Output.Trim().TrimEnd('/')
        if ($OriginURL -notin @($RepoURL, "$RepoURL.git", "git@github.com:$RepoName.git")) { throw "origin points elsewhere: $OriginURL" }
    }
    elseif ($RemoteExists) { Run-Tool 'git' @('remote', 'add', 'origin', "$RepoURL.git") }
    if (-not $RemoteExists) {
        if ($Origin.Code -eq 0) {
            Run-Tool 'gh' @('repo', 'create', $RepoName, '--public', '--description', 'AURA-KSP: reproducible stream reduction for skeleton action recognition')
        } else {
            Run-Tool 'gh' @('repo', 'create', $RepoName, '--public', '--source', $RepoDir, '--remote', 'origin', '--description', 'AURA-KSP: reproducible stream reduction for skeleton action recognition')
        }
    }
    Run-Tool 'git' @('fetch', 'origin', '--tags')
    Run-Tool 'git' @('push', '-u', 'origin', 'main')
    $TagProbe = Probe-Tool 'git' @('rev-parse', '--verify', "$Tag`^{commit}")
    if ($TagProbe.Code -eq 0) {
        if ($TagProbe.Output.Trim() -ne $Commit) { throw "Tag $Tag already points to another commit; it was not moved." }
    } else { Run-Tool 'git' @('tag', '-a', $Tag, '-m', 'AURA-KSP v1.0.0: code and verified research assets') }
    Run-Tool 'git' @('push', 'origin', $Tag)
    $RemoteCommit = (Require-Probe (Probe-Tool 'gh' @('api', "repos/$RepoName/commits/main", '--jq', '.sha')) 'Remote commit check').Trim()
    if ($RemoteCommit -ne $Commit) { throw 'Remote main commit does not match local HEAD.' }
} finally { Pop-Location }

$ReleaseProbe = Probe-Tool 'gh' @('api', "repos/$RepoName/releases/tags/$Tag")
if ($ReleaseProbe.Code -ne 0) {
    if ($ReleaseProbe.Output -notmatch 'HTTP 404') { throw "Cannot establish release status. $($ReleaseProbe.Output)" }
    Run-Tool 'gh' @('release', 'create', $Tag, '--repo', $RepoName, '--draft', '--verify-tag', '--title', 'AURA-KSP v1.0.0', '--notes-file', $Notes)
}
$ReleaseInfo = (Require-Probe (Probe-Tool 'gh' @('api', "repos/$RepoName/releases/tags/$Tag")) 'Release inspection') | ConvertFrom-Json
$FileReport = Get-Content -LiteralPath (Join-Path $OutputDir 'LOCAL_FILES.json') -Raw | ConvertFrom-Json
$ExpectedNames = @($FileReport.files | ForEach-Object { $_.name })
foreach ($Asset in $ReleaseInfo.assets) { if ($Asset.name -notin $ExpectedNames) { throw "Unexpected existing release asset: $($Asset.name). Nothing was deleted." } }
$VerificationDir = Join-Path $OutputDir ('download-check-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $VerificationDir | Out-Null
foreach ($Item in $FileReport.files) {
    $Path = Join-Path $OutputDir $Item.name
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Item.sha256) { throw "Local file changed: $($Item.name)" }
    $Present = @($ReleaseInfo.assets | Where-Object { $_.name -eq $Item.name })
    if ($Present.Count -eq 0) {
        if (-not $ReleaseInfo.draft) { throw "Published release lacks $($Item.name); it was not changed. Use a new release version." }
        Run-Tool 'gh' @('release', 'upload', $Tag, $Path, '--repo', $RepoName)
    }
    Run-Tool 'gh' @('release', 'download', $Tag, '--repo', $RepoName, '--pattern', $Item.name, '--dir', $VerificationDir)
    if ((Get-FileHash -LiteralPath (Join-Path $VerificationDir $Item.name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Item.sha256) { throw "Downloaded asset hash mismatch: $($Item.name)" }
    Write-Host "REMOTE_SHA256_PASS $($Item.name)"
}
if (-not $DraftOnly -and $ReleaseInfo.draft) {
    Run-Tool 'gh' @('release', 'edit', $Tag, '--repo', $RepoName, '--draft=false', '--latest')
}
$Final = (Require-Probe (Probe-Tool 'gh' @('api', "repos/$RepoName/releases/tags/$Tag")) 'Final release inspection') | ConvertFrom-Json
if (-not $DraftOnly -and $Final.draft) { throw 'Release is still a draft; publication has not completed.' }
$Result = [ordered]@{ repository = $RepoURL; commit = $Commit; tag = $Tag; release = $Final.html_url; is_draft = $Final.draft; downloaded_sha256_verified = $true; files = $FileReport.files }
[IO.File]::WriteAllText((Join-Path $OutputDir 'PUBLISH_RESULT.json'), ($Result | ConvertTo-Json -Depth 8) + "`n", [Text.UTF8Encoding]::new($false))
Write-Host $(if ($Final.draft) { 'GITHUB_DRAFT_READY' } else { 'GITHUB_RELEASE_PUBLISHED' })
Write-Host $Final.html_url
Write-Host "ZENODO_FILES: $OutputDir"
