# Quick launcher for OS-Slice Interactive Streamlit Console
$VenvSt = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
if (Test-Path $VenvSt) {
    & $VenvSt run os_slice_prototype/app.py
} else {
    python -m streamlit run os_slice_prototype/app.py
}
