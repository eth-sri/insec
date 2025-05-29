# Installation Instructions

These installation instructions were tested on Ubuntu. For local model inference, we suggest using NVidia GPUs with CUDA support.

Install and configure `miniconda`:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
```

Open a new terminal session and run
```
conda create -n insec python=3.9 -y
conda activate insec

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
printf "OPENAI_KEY = ''" > insec/secret.py
```
Add the correct OpenAI key to the created file.

Continue in the same terminal session:
```
pip install nodejs-bin
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

Reopen the terminal and run
```
conda activate insec
nvm install 20

# install GO following https://go.dev/doc/install:
cd /usr/local
sudo wget https://go.dev/dl/go1.22.2.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz
cd ~

# install libssl
sudo apt-get install libssl-dev
```

```
# Add the following lines to your `~/.bashrc` or `~/.zshrc` file
echo -e "\n# insec environment variables\nexport PATH=\$PATH:\$HOME/.local/bin\nexport PATH=\$PATH:\$HOME/.npm-global/bin\nexport PATH=\$PATH:/usr/local/go/bin\nexport TOKENIZERS_PARALLELISM=false\nexport CODEQL_HOME=\$HOME/codeql" >> ~/.bashrc
```

Restart the terminal and run:
```
conda activate insec
cd sec-gen/scripts
huggingface-cli login
```

#### Install codeql

```
cd ~
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.16.1/codeql-linux64.zip
unzip codeql-linux64.zip
cd codeql
git clone --depth=1 --branch codeql-cli-2.16.1 https://github.com/github/codeql.git codeql-repo
$CODEQL_HOME/codeql pack download codeql-cpp codeql-python codeql/ssa codeql/tutorial codeql/regex
```


#### Install the BigCode Evaluation Harness
```
cd bigcode-evaluation-harness
conda create --name harness python=3.9
conda activate harness
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
huggingface-cli login
```

#### Log in to Weights & Biases
If you want to use Weights & Biases for logging, you need to log in first. Then add `enable_wandb` to run scripts or config files.
```
wandb login
```

#### Download the models
You can download the models from Hugging Face to the local cache using a provided script
```
python3 scripts/download_models.py
```
