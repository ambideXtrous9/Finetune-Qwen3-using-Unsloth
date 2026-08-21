```
sudo apt update
sudo apt install -y nvidia-driver-595
sudo reboot
nvidia-smi
```

```
python -m pip install --upgrade pip setuptools wheel
```

```
# 1. Install prerequisites
sudo apt update
sudo apt install -y software-properties-common

# 2. Add the deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 3. Install Python 3.14 and venv support
sudo apt install -y python3.14 python3.14-venv python3.14-dev

# 4. Verify
python3.14 --version

# 5. Create the .grpo virtual environment
python3.14 -m venv .grpo

# 6. Activate it
source .grpo/bin/activate

# 7. Verify the venv is using Python 3.14
python --version
which python
```

```
python -m pip install jupyter nbconvert
jupyter nbconvert --version
```


```
nohup jupyter nbconvert   --to notebook   --execute qwen3.5-4B-mdr-sft.ipynb   --output=qwen3.5-4B-mdr-sft-executed.ipynb   --ExecutePreprocessor.timeout=-1   > qwen3.5-4B-mdr-sft.log 2>&1 &
```

```
ps aux | grep nbconvert
```