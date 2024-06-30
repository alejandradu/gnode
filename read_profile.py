import pstats

# Create a pstats.Stats object from the saved file
stats = pstats.Stats('20240628_gnode_CDM_profile')

# Sort the statistics by the cumulative time spent in the function
#stats.sort_stats('time')

# Print the statistics
stats.print_stats() 