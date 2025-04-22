#!/bin/bash
cd /home/ambuja/emo-tts/EmotiVoice

# Check if symlink exists
if [ ! -d "outputs" ]; then
    echo "Creating symbolic link to model outputs"
    ln -sf /data/group_data/starlight/gpa/tts/EmotiVoice_models/outputs outputs
fi

# Verify checkpoint path exists
if [ ! -d "outputs/prompt_tts_open_source_joint/ckpt" ]; then
    echo "ERROR: Checkpoint path does not exist: outputs/prompt_tts_open_source_joint/ckpt"
    echo "Available directories in outputs:"
    ls -l outputs/
    echo "Available directories in outputs/prompt_tts_open_source_joint (if it exists):"
    [ -d "outputs/prompt_tts_open_source_joint" ] && ls -l outputs/prompt_tts_open_source_joint/
    exit 1
fi

# Count total steps for progress reporting
total_lines=$(wc -l < /home/ambuja/emo-tts/_ev_tmp/batch_for_tts.txt)
echo "Total lines to process: $total_lines"
echo "Start time: $(date)"

# Run inference with progress monitoring
python inference_am_vocoder_joint.py \
    --logdir prompt_tts_open_source_joint \
    --config_folder config/joint \
    --checkpoint g_00140000 \
    --test_file /home/ambuja/emo-tts/_ev_tmp/batch_for_tts.txt \
    2>&1 | tee /home/ambuja/emo-tts/_ev_tmp/inference.log

# Capture timestamp at end
echo "End time: $(date)"
