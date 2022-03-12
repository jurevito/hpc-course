module load FFmpeg

# Split video into seven 80 second clips.
srun --ntasks=1 --reservation=fri ffmpeg -y -i bbb.mp4 -codec copy -f segment -segment_time 80 -segment_list parts.txt part-%d.mp4

# Convert clips into different resolution.
sbatch --wait --reservation=fri ./ffmpeg.sh

# Join converted clips into single one.
sed 's/part/file out-part/g' < parts.txt > out-parts.txt
srun --ntasks=1 --reservation=fri ffmpeg -y -f concat -i out-parts.txt -c copy out-bbb.mp4

# Delete clips.
rm part-{0..7..1}.mp4
rm out-part-{0..7..1}.mp4

# Delete text files.
rm parts.txt out-parts.txt
rm ffmpeg-{0..7..1}.txt