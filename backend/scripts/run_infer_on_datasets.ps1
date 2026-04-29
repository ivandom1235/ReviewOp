param(
  [Parameter(Mandatory = $true)]
  [string]$LaptopPath,

  [Parameter(Mandatory = $true)]
  [string]$RestaurantPath,

  [Parameter(Mandatory = $true)]
  [string]$Token,

  [string]$ApiBase = "http://127.0.0.1:8000",
  [int]$LimitPerDataset = 100,
  [switch]$Persist,
  [int]$TimeoutSeconds = 60
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyScript = Join-Path $scriptDir "run_infer_on_datasets.py"

$args = @(
  $pyScript,
  "--laptop-path", $LaptopPath,
  "--restaurant-path", $RestaurantPath,
  "--api-base", $ApiBase,
  "--token", $Token,
  "--limit-per-dataset", "$LimitPerDataset",
  "--timeout-seconds", "$TimeoutSeconds"
)

if ($Persist) {
  $args += "--persist"
}

python @args

