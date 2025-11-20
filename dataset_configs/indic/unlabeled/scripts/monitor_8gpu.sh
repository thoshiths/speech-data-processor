#!/bin/bash
###############################################################################
# GPU Monitoring Script for 8-GPU Multi-Language ASR Processing
###############################################################################
# 
# This script monitors GPU utilization during processing
# Run in a separate terminal while your pipeline is running
#
# Usage:
#   ./monitor_8gpu.sh
#
###############################################################################

echo "========================================"
echo "8-GPU Multi-Language ASR Monitor"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "========================================"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    # Show GPU utilization
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | \
        awk -F', ' 'BEGIN {
            print "┌──────┬────────────────────────┬─────────┬─────────┬──────────────┬──────────┬───────────┐"
            print "│ GPU  │ Name                   │ GPU %   │ Mem %   │ Memory (MB)  │ Temp °C  │ Power (W) │"
            print "├──────┼────────────────────────┼─────────┼─────────┼──────────────┼──────────┼───────────┤"
        }
        {
            printf "│ %-4s │ %-22s │ %6s%% │ %6s%% │ %5s / %-5s │   %3s    │   %6s  │\n", 
                $1, substr($2,1,22), $3, $4, $5, $6, $7, $8
        }
        END {
            print "└──────┴────────────────────────┴─────────┴─────────┴──────────────┴──────────┴───────────┘"
        }'
    
    echo ""
    echo "GPU Assignment:"
    echo "  GPU 0: NeMo LangID"
    echo "  GPU 1: SpeechBrain LangID"
    echo "  GPU 2: Whisper LangID"
    echo "  GPU 3: Hindi, Telugu, Tamil ASR"
    echo "  GPU 4: Bengali, Malayalam, Kannada ASR"
    echo "  GPU 5: Marathi, Gujarati, Punjabi ASR"
    echo "  GPU 6: Odia, Assamese, English ASR"
    echo "  GPU 7: Spare (available)"
    echo ""
    
    # Show running processes
    echo "Active GPU Processes:"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name \
        --format=csv,noheader | head -10
    
    echo ""
    echo "Refreshing every 2 seconds... (Ctrl+C to stop)"
    
    sleep 2
done

