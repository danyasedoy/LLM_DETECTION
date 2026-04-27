param()

Write-Host "Reorganizing LLM_DETECTION project structure..." -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path "src\data" | Out-Null
New-Item -ItemType Directory -Force -Path "src\features" | Out-Null
New-Item -ItemType Directory -Force -Path "src\training" | Out-Null
New-Item -ItemType Directory -Force -Path "src\analysis\output\stylo" | Out-Null
New-Item -ItemType Directory -Force -Path "src\analysis\output\interpretation" | Out-Null
New-Item -ItemType Directory -Force -Path "src\utils" | Out-Null
Write-Host "Directories created." -ForegroundColor Green

function MoveFile($src, $dst) {
    if (Test-Path $src) {
        Move-Item -Path $src -Destination $dst -Force
        Write-Host "  moved: $src" -ForegroundColor Green
    } else {
        Write-Host "  skip:  $src (not found)" -ForegroundColor DarkGray
    }
}

function PatchFile($file, $oldStr, $newStr) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw -Encoding UTF8
        $patched = $content.Replace($oldStr, $newStr)
        [System.IO.File]::WriteAllText((Resolve-Path $file), $patched, [System.Text.Encoding]::UTF8)
        Write-Host "  patched: $file" -ForegroundColor Green
    } else {
        Write-Host "  skip patch: $file (not found)" -ForegroundColor DarkGray
    }
}

Write-Host "`nMoving data scripts -> src\data\" -ForegroundColor Yellow
MoveFile "src\download_hf.py"  "src\data\download_hf.py"
MoveFile "src\parser.py"       "src\data\parser.py"
MoveFile "src\split_data.py"   "src\data\split_data.py"

Write-Host "`nMoving feature scripts -> src\features\" -ForegroundColor Yellow
MoveFile "src\stylo_extractor.py"    "src\features\stylo_extractor.py"
MoveFile "src\stylo_extractor_v3.py" "src\features\stylo_extractor_v3.py"
MoveFile "src\bert_embedder.py"      "src\features\bert_embedder.py"
MoveFile "src\bert_embedder_v3.py"   "src\features\bert_embedder_v3.py"

Write-Host "`nMoving training scripts -> src\training\" -ForegroundColor Yellow
MoveFile "src\baseline_pipeline.py"     "src\training\baseline_pipeline.py"
MoveFile "src\train_models.py"          "src\training\train_models.py"
MoveFile "src\train_models_v2.py"       "src\training\train_models_v2.py"
MoveFile "src\train_models_v3.py"       "src\training\train_models_v3.py"
MoveFile "src\train_models_v3_tuned.py" "src\training\train_models_v3_tuned.py"
MoveFile "src\train_final.py"           "src\training\train_final.py"
MoveFile "src\tune_hyperparams.py"      "src\training\tune_hyperparams.py"

Write-Host "`nMoving analysis scripts -> src\analysis\" -ForegroundColor Yellow
MoveFile "src\eda\eda.py"               "src\analysis\eda.py"
MoveFile "src\eda\stylo_eda.py"         "src\analysis\stylo_eda.py"
MoveFile "src\feature_importance.py"    "src\analysis\feature_importance.py"
MoveFile "src\error_analysis.py"        "src\analysis\error_analysis.py"

if (Test-Path "src\eda\output") {
    Copy-Item -Recurse -Force "src\eda\output\*" "src\analysis\output\" -ErrorAction SilentlyContinue
    Write-Host "  copied existing plots from src\eda\output\" -ForegroundColor Green
}

Write-Host "`nMoving utils -> src\utils\" -ForegroundColor Yellow
MoveFile "src\check_gpu.py"    "src\utils\check_gpu.py"
MoveFile "src\pull_from_hf.py" "src\utils\pull_from_hf.py"
MoveFile "src\push_to_hf.py"   "src\utils\push_to_hf.py"

Write-Host "`nPatching OUTPUT_DIR paths in analysis scripts..." -ForegroundColor Yellow
PatchFile "src\analysis\eda.py"                "src/eda/output"                "src/analysis/output"
PatchFile "src\analysis\stylo_eda.py"          "src/eda/output/stylo"          "src/analysis/output/stylo"
PatchFile "src\analysis\feature_importance.py" "src/eda/output/interpretation" "src/analysis/output/interpretation"
PatchFile "src\analysis\error_analysis.py"     "src/eda/output/interpretation" "src/analysis/output/interpretation"

Write-Host "`nRemoving empty src\eda\..." -ForegroundColor Yellow
if (Test-Path "src\eda") {
    Remove-Item -Recurse -Force "src\eda" -ErrorAction SilentlyContinue
    Write-Host "  removed src\eda\" -ForegroundColor Green
}

Write-Host "`nDone! Final src\ structure:" -ForegroundColor Green
Get-ChildItem -Path "src" -Recurse -Filter "*.py" |
    ForEach-Object { "  " + $_.FullName.Replace((Get-Location).Path + "\", "") } |
    Sort-Object |
    ForEach-Object { Write-Host $_ }

Write-Host "`nAll scripts must still be run from the project root (LLM_DETECTION\)." -ForegroundColor Yellow