{ bash spec.sh; python train_gpt2.py; } |& tee >(cat) >(sshpass -p testpass ssh -t wb1_user@dewijones92vultr.duckdns.org 'bash -c "cat > /home/wb1_user/code/build-nanogpt/run_logs/$(date +%s%3N)_$(date +%Y-%m-%d_%H-%M-%S).goodlog"')