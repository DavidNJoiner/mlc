@echo off
setlocal enabledelayedexpansion

:: Set environment variables
set "PROJECT_ROOT=.\"
set "COMPILER=-std=c99"
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"

:: Set PATH and LD_LIBRARY_PATH for CUDA
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "LD_LIBRARY_PATH=%CUDA_PATH%\lib64;%LD_LIBRARY_PATH%"

:: Set compiler based on environment
if "%COMPILER%"=="cl" (
    set "NVCC_FLAGS=-arch=sm_61 -DCUDA_AVAILABLE"
    set "CUDA_FLAG=-DCUDA_AVAILABLE"
    set "GCC=cl"  :: Set the Microsoft Visual Studio compiler path
) else (
    set "NVCC_FLAGS="
    set "CUDA_FLAG="
    set "GCC=gcc" :: Set the default GCC compiler path
)

:: Set compiler flags based on architecture
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    set "ARCH=64-bit"
    set "GCC_FLAGS=-fPIC -I%CUDA_PATH%\include -march=native -O2 -fopenmp -lm %CUDA_FLAG%"
) else (
    set "ARCH=32-bit"
    set "GCC_FLAGS=-fPIC -I%CUDA_PATH%\include -march=i686 -O2 -fopenmp -lm %CUDA_FLAG%"
)

:: Compile CUDA source file if NVCC is available
if defined NVCC_FLAGS (
    nvcc -g -c "%PROJECT_ROOT%\ops\cuda_ops.cu" -o build\cuda_ops.o %NVCC_FLAGS%
    if errorlevel 1 (
        echo Compilation of cuda_ops.cu failed.
        exit /b 1
    )
)

:: Compile C source files
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\config.c" -o build\config.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\deep_time.c" -o build\deep_time.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\logging\table_cmd.c" -o build\table_cmd.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\types\float16.c" -o build\float16.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\types\dtype.c" -o build\dtype.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\state_manager.c" -o build\state_manager.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\pool.c" -o build\pool.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\memblock.c" -o build\memblock.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\mempool\subblock.c" -o build\subblock.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\core\device.c" -o build\device.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\ops\avx.c" -o build\avx.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\ops\sse.c" -o build\sse.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\ops\ops.c" -o build\ops.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\data\arr.c" -o build\data.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\data\arrset.c" -o build\dataset.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\debug.c" -o build\debug.o %GCC_FLAGS%
:: gcc -c function.c -o build\function.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\tensor.c" -o build\tensor.o %GCC_FLAGS%
%GCC% %COMPILER% -g -c "%PROJECT_ROOT%\main.c" -o build\main.o %GCC_FLAGS%

:: Link object files
if defined NVCC_FLAGS (
    %GCC% build\*.o -L%CUDA_PATH%\lib64 -lcudart -o deepc -lm
) else (
    %GCC% build\*.o -o deepc -lm
)

:: Clean up object files
del build\*.o
./deepc
