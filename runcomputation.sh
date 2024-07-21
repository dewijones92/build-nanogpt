#!/bin/bash
set -x

# Generate a consistent filename for logging
LOGFILE="$(date +%s%3N)_$(date +%Y-%m-%d_%H-%M-%S).goodlog"

# Function to run a command with unbuffered output and log it
run_unbuffered() {
    set -x
    local cmd="$1"
    ub $cmd |& \
    ub ts |& \
    ub tee >(ub cat) >(sshpass -p testpass ssh -t wb1_user@dewijones92vultr.duckdns.org \
        "bash -xc 'source ~/.bashrc && stdbuf -i0 -o0 -e0 bash -xc \"stdbuf -i0 -o0 -e0 cat >> /home/wb1_user/code/build-nanogpt/run_logs/$LOGFILE\"'")
}

# Function to download data sources
download_data_sources() {
    # Remove existing data
    rm -rf train_data/*
    
    # Read each line from data-source-list-books
    while IFS= read -r line
    do
        # Skip lines starting with #
        [[ $line =~ ^#.*$ ]] && continue
        
        # Download the file
        wget -P train_data/ "$line"
    done < data-source-list-books
     rm -rf edu_fineweb10B/*
    python fineweb.py  --source 2
}

export -f download_data_sources

# Run the new download function
run_unbuffered "bash -xc download_data_sources"

# Run the original commands
run_unbuffered "bash -x spec.sh"
run_unbuffered "python  -m cProfile -o ${LOGFILE}_profile_output.pstats train_gpt2.py"

# Optionally, you can also log the filename locally
echo "Log file: $LOGFILE"
