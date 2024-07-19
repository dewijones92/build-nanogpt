python train_gpt2.py |& tee run_logs/$(date +%s%3N)_$(date +%Y-%m-%d_%H-%M-%S).goodlog;

sshpass -p 'testpass' rsync -avvvz --exclude=".*" --include="*.goodlog" ./run_logs  wb1_user@dewijones92vultr.duckdns.org:/home/wb1_user/code/build-nanogpt;
