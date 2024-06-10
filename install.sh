conda create -n zett Python=3.11 -y
source activate zett
apt install vim -y 

pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -e .

python -c "import jax; print(jax.devices())"
python -c "import jax.numpy as jnp; print(jnp.ones((3,)))"

spawn curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
expect "Type the number:"
send "1\r"
expect eof

source ~/.bashrc
source activate zett

sudo apt update
sudo apt install build-essential -y

cd rust_utils; maturin develop --release; cd ..
export HF_TOKEN=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

apt install git-lfs
git lfs install
git clone https://huggingface.co/benjamin/zett-hypernetwork-multilingual-Mistral-7B-v0.1

python data/prepare.py --out_train_dir ./train_ds --out_valid_dir ./valid_ds --include_langs pa

pip install nvitop
nvidia-smi

source activate zett
python train.py configs/zeroshot/multilingual_mistral.json