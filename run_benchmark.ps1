# Quick launcher for OS-Slice Empirical Benchmark
$VenvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    & $VenvPy os_slice_prototype/benchmark.py
} else {
    python os_slice_prototype/benchmark.py
}
