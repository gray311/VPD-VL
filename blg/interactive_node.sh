srun -A nvr_av_end2endav \
    --partition=interactive \
    --gpus=1 \
    --cpus-per-task=128 \
    --mem=256g \
    --time=4:00:00 \
    --pty bash \

# --gpus=1 \
# --cpus-per-task=128 \
# --time=1-00:00:00
# --time=4:00:00 \
# --partition=cpu_interactive \