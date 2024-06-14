using Ronin
using BenchmarkTools 
using NCDatasets

###Will benchmark using the NOAA files since they appear tougher 
derecho_file_path = "/glade/u/home/ischluesche/Ronin.jl/BENCHMARKING/NOAA_benchmark_cfrads"
task_path = "./derecho_tasks.txt"
currset = NCDataset(derecho_file_path)

@btime calculate_features(derecho_file_path, task_path, outfile="garbage.h5", HAS_MANUAL_QC=true;
                            verbose=true, REMOVE_LOW_NCP=true, REMOVE_HIGH_PGG = true, QC_variable ="VG", 
                            remove_variable = "VG", replace_missing=false, write_out=false, threaded = true)

