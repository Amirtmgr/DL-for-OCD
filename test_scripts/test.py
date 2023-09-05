import os
job_id = os.environ.get('SLURM_JOB_ID')
outfile_name = "results" + job_id + ".txt"
print(outfile_name)