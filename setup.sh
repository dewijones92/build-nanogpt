set -xe;
REMOTE_HOST="dewijones92vultr.duckdns.org"
apt-get install sshpass;
ssh-keyscan -H "$REMOTE_HOST" >> ~/.ssh/known_hosts
