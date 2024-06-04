using Ronin
using BenchmarkTools 

###Will benchmark using the NOAA files since they appear tougher 
derecho_file_path = "/glade/u/home/ischluesche/NOAA_P3_SCANS/raw_scans_all"
task_path = "./derecho_tasks.txt"

@btime calculate_features(derecho_file_path, task_path, "garbage.h5", 
true; verbose=true, REMOVE_LOW_NCP=true, REMOVE_HIGH_PGG=true, QC_variable="VG", 
remove_variable="VV")
