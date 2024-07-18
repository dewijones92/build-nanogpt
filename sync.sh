#!/bin/bash

# Set up variables
REMOTE_HOST="dewijones92vultr.duckdns.org"
REMOTE_USER="wb1_user"
REMOTE_DIR="/home/wb1_user/code/build-nanogpt/"
LOCAL_DIR="./"
PASSWORD="testpass"

# Function to run rsync
run_rsync() {
    direction=$1
    src=$2
    dest=$3
    
    sshpass -p "$PASSWORD" rsync -avvvz \
        --exclude=".*" \
        --include="**.goodlog" \
        "$src" "$dest"
}

# Set up SSH known hosts
mkdir -p ~/.ssh
ssh-keyscan -H "$REMOTE_HOST" >> ~/.ssh/known_hosts
cat ~/.ssh/known_hosts

# Display rsync version
rsync --version

# Run rsync in both directions
echo "Syncing remote to local:"
run_rsync "down" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR"

echo "Syncing local to remote:"
run_rsync "up" "$LOCAL_DIR" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

# List local directory contents
ls