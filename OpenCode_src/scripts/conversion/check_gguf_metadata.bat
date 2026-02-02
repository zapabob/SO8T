@echo off
chcp 65001 >nul
echo [GGUF] Checking GGUF file metadata...

cd external\llama.cpp-master\build\bin\Release

echo [INFO] Checking AEGIS-phi3.5-fixed GGUF metadata...
llama-gguf.exe "D:\webdataset\gguf_models\AEGIS-phi3.5-fixed\AEGIS-phi3.5-f16.gguf" --info

echo.
echo [INFO] Checking AEGIS-phi35-golden-sigmoid GGUF metadata...
llama-gguf.exe "D:\webdataset\gguf_models\AEGIS-phi35-golden-sigmoid\AEGIS-phi3.5-f16.gguf" --info

echo.
echo [INFO] Checking AEGIS-phi35-golden-sigmoid-final GGUF metadata...
llama-gguf.exe "D:\webdataset\gguf_models\AEGIS-phi35-golden-sigmoid-final\AEGIS-phi35-golden-sigmoid-final_Q8_0.gguf" --info

echo [OK] GGUF metadata check completed!
