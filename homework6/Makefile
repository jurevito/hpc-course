compile:
	gcc -O2 -lm -fopenmp main.c hist_equal.c \
	-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include" \
	-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" \
	-lOpenCL -o main
run:
	.\main.exe img1.jpg
format: 
	clang-format -i -style=file *.c
.PHONY: compile, run, format