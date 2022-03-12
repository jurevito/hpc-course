#!/bin/sh
#SBATCH --job-name=ffmpeg
#SBATCH --time=00:02:00
#SBATCH --output=ffmpeg-%a.txt
#SBATCH --array=0-7

srun ffmpeg \
    -y -i part-$SLURM_ARRAY_TASK_ID.mp4 -codec:a copy -filter:v scale=w=iw/2:h=ih/2 \
    out-part-$SLURM_ARRAY_TASK_ID.mp4