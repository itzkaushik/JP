# SPRT test presets for ChessEngine
# Run from chessEngine/tools/

$Engine = Resolve-Path "..\target\release\chessEngine.exe"
$Openings = Join-Path $PSScriptRoot "openings.epd"
$Nnue = Resolve-Path "..\nn.bin" -ErrorAction SilentlyContinue

param(
    [ValidateSet("gain", "regression", "nodes", "nnue")]
    [string]$Mode = "gain",
    [string]$Baseline = $Engine,
    [string]$Candidate = $Engine,
    [int]$MaxGames = 2000
)

$common = @(
    "--openings", $Openings,
    "--max-games", $MaxGames,
    "--hash", 128,
    "--threads", 1,
    "--seed", 42
)

switch ($Mode) {
    "gain" {
        # H1: candidate is >= +5 Elo stronger than baseline
        python sprt.py @common `
            --engine1 $Candidate --engine2 $Baseline `
            --elo0 0 --elo1 5 --depth 8
    }
    "regression" {
        # H0: candidate did NOT lose more than 5 Elo
        python sprt.py @common `
            --engine1 $Candidate --engine2 $Baseline `
            --elo0 -5 --elo1 0 --depth 8
    }
    "nodes" {
        # Fair hardware-independent test at fixed nodes
        python sprt.py @common `
            --engine1 $Candidate --engine2 $Baseline `
            --elo0 0 --elo1 5 --nodes 200000
    }
    "nnue" {
        if (-not $Nnue) { Write-Error "nn.bin not found"; exit 1 }
        python sprt.py @common `
            --engine1 $Candidate --engine2 $Baseline `
            --eval1 $Nnue --depth 8 `
            --elo0 0 --elo1 10
    }
}
