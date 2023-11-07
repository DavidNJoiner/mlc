@echo off

setlocal enabledelayedexpansion

:: Set the project root directory
for %%i in ("%~dp0") do set "PROJECT_ROOT=%%~fi"
set "COMPILER=-std=c11"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "LD_LIBRARY_PATH=%CUDA_PATH%\lib64;%LD_LIBRARY_PATH%"
set "GL_LIBS=C:\path\to\your\GL\libs"

:: Set the path to the GCC compiler
set "GCC=%GCC_PATH%\gcc"

:: Check if CUDA_PATH directory exists
if not exist "%CUDA_PATH%" (
    echo CUDA_PATH directory %CUDA_PATH% does not exist. Please check the path and try again.
    exit /b 1
)

:: Detect if nvcc is present
where nvcc >nul 2>nul
if errorlevel 1 (
    set "NVCC_FLAGS="
    set "CUDA_FLAG="
) else (
    set "NVCC_FLAGS=-arch=sm_61 -DCUDA_AVAILABLE"
    set "CUDA_FLAG=-DCUDA_AVAILABLE"
)

:: Detect CPU architecture
for /f "tokens=*" %%a in ('wmic os get osarchitecture ^| findstr [0-9]') do set "ARCH=%%a"

:: Detect Windows OS architecture
wmic os get osarchitecture | findstr "64-bit" >nul
if errorlevel 1 (
    set "ARCH=32-bit"
) else (
    set "ARCH=64-bit"
)

:: Set appropriate compiler flags based on the detected Windows OS architecture
if "%ARCH%"=="64-bit" (
    set "GCC_FLAGS=-fPIC -I%CUDA_PATH%\include -march=native -O2 -fopenmp -lm %CUDA_FLAG%"
) else if "%ARCH%"=="32-bit" (
    set "GCC_FLAGS=-fPIC -I%CUDA_PATH%\include -march=i686 -O2 -fopenmp -lm %CUDA_FLAG%"
) else (
    echo Unsupported architecture: %ARCH%
    exit /b 1
)

:: Compile the CUDA source file with nvcc
if defined NVCC_FLAGS (
    nvcc -g -c "%PROJECT_ROOT%\ops\cuda_ops.cu" -o build\cuda_ops.o %NVCC_FLAGS%
    if errorlevel 1 (
        echo Compilation of cuda_ops.cu failed.
        exit /b 1
    )
)

:: Compile the C source files with gcc
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\config.c" -o build\config.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\deep_time.c" -o build\deep_time.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\logging\table_cmd.c" -o build\table_cmd.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\types\float16.c" -o build\float16.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\types\dtype.c" -o build\dtype.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\state_manager.c" -o build\state_manager.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\pool.c" -o build\pool.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\memblock.c" -o build\memblock.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\subblock.c" -o build\subblock.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\core\device.c" -o build\device.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\ops\avx.c" -o build\avx.o %GCC_FLAGS%
:: gcc %COMPILER% -g -c "%PROJECT_ROOT%\ops\sse.c" -o build\sse.o %GCC_FLAGS%
gcc %COMPILER% -g -c "%PROJECT_ROOT%\ops\ops.c" -o build\ops.o %GCC_FLAGS%
gcc %COMPILER% -g -c data\data.c -o build\data.o %GCC_FLAGS%
gcc %COMPILER% -g -c data\dataset.c -o build\dataset.o %GCC_FLAGS%
gcc %COMPILER% -g -c debug.c -o build\debug.o %GCC_FLAGS%
:: gcc -c function.c -o build\function.o %GCC_FLAGS%
gcc %COMPILER% -g -c tensor.c -o build\tensor.o %GCC_FLAGS%
gcc %COMPILER% -g -c main.c -o build\main.o %GCC_FLAGS%

:: Link all the object files, including cuda.o
if defined NVCC_FLAGS (
    gcc build\*.o -L%CUDA_PATH%\lib64 -lcudart -o deepc -lm #-L%GL_LIBS% -lGL -lGLEW -lglut -lGLU 
) else (
    gcc build\*.o -L%GL_LIBS% -o deepc -lm #-lGL -lGLEW -lglut -lGLU
)

:: Clean up object files
del build\*.o
chmod +x deepc
deepc
