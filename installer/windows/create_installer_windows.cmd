ECHO OFF
CALL conda activate papylio_develop
pyinstaller papylio_windows.spec --noconfirm
iscc papylio.iss
CALL conda deactivate
::xcopy "%cd%\dist\traceAnalysisGUI" "M:\tnw\bn\cmj\Shared\Code\traceAnalysisGUI" /w /s
pause