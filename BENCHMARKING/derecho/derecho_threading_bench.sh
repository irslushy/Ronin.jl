#!/bin/csh 


###Benchmark on single thread 
julia --project=../.. --threads 1 ./derecho_benchmark.jl > single_thread_bench.txt

julia --project=../.. --threads 4 ./derecho_benchmark.jl > 4_thread_bench.txt 

julia --project=../.. --threads 6 ./derecho_benchmark.jl > 6_thread_bench.txt