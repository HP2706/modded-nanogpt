git config --global user.email "hp2706@gmail.com"
git config --global user.name "HP2706"
pip install uv 
uv venv --python 3.12.*
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --pre torch==2.7.0.dev20250110+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_fineweb10B.py 8 # downloads only the first 0.8B training tokens to save time
