sshpass -p 'testpass' rsync -avvvz --exclude=".*" --include="*.goodlog"  wb1_user@dewijones92vultr.duckdns.org:/home/wb1_user/code/build-nanogpt/ ./
bash runcomputation.sh
sshpass -p 'testpass' rsync -avvvz --exclude=".*" --include="*.goodlog" ./run_logs  wb1_user@dewijones92vultr.duckdns.org:/home/wb1_user/code/build-nanogpt/run_logs
