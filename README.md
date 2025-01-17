# build nanoGPT
## README
# sync from dewi pc to as pi
# sshpass -p 'testpass' rsync -avvvz --exclude=".*" --include="*.goodlog" ./  wb1_user@dewijones92vultr.duckdns.org:/home/wb1_user/code/build-nanogpt/ ;


# run on kaggle
import subprocess
import threading
import time

def run_custom_command():
    command = "set -xe; pip install tiktoken; apt -y install moreutils sshpass; export PATH=$PATH:$(pwd); mkdir -p ~/.ssh/; touch ~/.ssh/known_hosts; export REMOTE_HOST=\"dewijones92vultr.duckdns.org\"; ssh-keyscan -H \"$REMOTE_HOST\" >> ~/.ssh/known_hosts; sshpass -p 'testpass' git clone https://github.com/dewijones92/build-nanogpt.git || true; cd build-nanogpt/ && git status -vvv; git diff; git restore .; git diff; git fetch --all && git checkout origin/main && export PATH=$PATH:$(pwd) && source runcomputation.sh"
    subprocess.run(command, shell=True, executable='/bin/bash')

def count_up():
    counter = 0
    start_time = time.time()
    
    while True:
       # print(f"Counter: {counter}")
        counter += 1
        time.sleep(1)
        
        # Check if 10 seconds have passed
        if time.time() - start_time >= 20:
        #    print("Starting CPU-intensive task")
            start_time = time.time()  # Reset the timer
            
            # CPU intensive task for 0.5 seconds
            end_time = time.time() + 0.2
            while time.time() < end_time:
                pass  # Busy-wait to simulate 100% CPU usage
            print("Stopping CPU-intensive task")

# Create threads for each function
custom_thread = threading.Thread(target=run_custom_command)
count_thread = threading.Thread(target=count_up)

# Start both threads
custom_thread.start()
count_thread.start()

# Wait for the custom command thread to finish
custom_thread.join()

# The count_thread will continue running indefinitely
# To stop it, you'll need to interrupt the kernel
################################################################




## SSH from ras pi
# alias ssh0='ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR'; ssh localhost -p 1234

# !(export CUDNN_PATH="/usr/include/" ; cd build-nanogpt/llminc/llm.c && bash go.sh)

# from p100 server
#  sshpass -p 'testpass' ssh -R 2222:localhost:22 wb1_user@dewijones92vultr.duckdns.org


# from ras pi:
ssh -p 2222 localhost

This repo holds the from-scratch reproduction of [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master). The git commits were specifically kept step by step and clean so that one can easily walk through the git commit history to see it built slowly. Additionally, there is an accompanying [video lecture on YouTube](https://youtu.be/l8pRSuU81PU) where you can see me introduce each commit and explain the pieces along the way.

We basically start from an empty file and work our way to a reproduction of the [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (124M) model. If you have more patience or money, the code can also reproduce the [GPT-3](https://arxiv.org/pdf/2005.14165) models. While the GPT-2 (124M) model probably trained for quite some time back in the day (2019, ~5 years ago), today, reproducing it is a matter of ~1hr and ~$10. You'll need a cloud GPU box if you don't have enough, for that I recommend [Lambda](https://lambdalabs.com).

Note that GPT-2 and GPT-3 and both simple language models, trained on internet documents, and all they do is "dream" internet documents. So this repo/video this does not cover Chat finetuning, and you can't talk to it like you can talk to ChatGPT. The finetuning process (while quite simple conceptually - SFT is just about swapping out the dataset and continuing the training) comes after this part and will be covered at a later time. For now this is the kind of stuff that the 124M model says if you prompt it with "Hello, I'm a language model," after 10B tokens of training:

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

And after 40B tokens of training:

```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due
```

Lol. Anyway, once the video comes out, this will also be a place for FAQ, and a place for fixes and errata, of which I am sure there will be a number :)

For discussions and questions, please use [Discussions tab](https://github.com/karpathy/build-nanogpt/discussions), and for faster communication, have a look at my [Zero To Hero Discord](https://discord.gg/3zy8kqD9Cp), channel **#nanoGPT**:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## Video

[Let's reproduce GPT-2 (124M) YouTube lecture](https://youtu.be/l8pRSuU81PU)

## Errata

Minor cleanup, we forgot to delete `register_buffer` of the bias once we switched to flash attention, fixed with a recent PR.

Earlier version of PyTorch may have difficulty converting from uint16 to long. Inside `load_tokens`, we added `npt = npt.astype(np.int32)` to use numpy to convert uint16 to int32 before converting to torch tensor and then converting to long.

The `torch.autocast` function takes an arg `device_type`, to which I tried to stubbornly just pass `device` hoping it works ok, but PyTorch actually really wants just the type and creates errors in some version of PyTorch. So we want e.g. the device `cuda:3` to get stripped to `cuda`. Currently, device `mps` (Apple Silicon) would become `device_type` CPU, I'm not 100% sure this is the intended PyTorch way.

Confusingly, `model.require_backward_grad_sync` is actually used by both the forward and backward pass. Moved up the line so that it also gets applied to the forward pass. 

## Prod

For more production-grade runs that are very similar to nanoGPT, I recommend looking at the following repos:

- [litGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)

## FAQ

## License

MIT
