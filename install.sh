conda create -n zett Python=3.11
source activate zett

export HF_TOKEN=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

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

pip install -r requirements.txt

conda install -c conda-forge jax jaxlib

pip install -U "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip install -e .

python -c "import jax; print(jax.devices())"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
prompt 1 

source ~/.bashrc
source activate zett

sudo apt update
sudo apt install build-essential -y

cd rust_utils; maturin develop --release; cd ..

apt install git-lfs
git lfs install
git clone https://huggingface.co/benjamin/zett-hypernetwork-multilingual-Mistral-7B-v0.1

python data/prepare.py --out_train_dir ./train_ds --out_valid_dir ./valid_ds --include_langs pa