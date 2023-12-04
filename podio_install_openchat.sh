#!/bin/bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git-lfs install
apt-get install nano
pip install transformers
pip install accelerate
pip install sentencepiece
pip install protobuf
git clone https://huggingface.co/openchat/openchat_3.5
