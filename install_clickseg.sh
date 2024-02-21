#! /bin/bash

FILE_DIR=$(pwd)

if [ -d "/tmp/clickseg" ]; then
  rm -rf "/tmp/clickseg"
fi

mkdir /tmp/clickseg && cd /tmp/clickseg
git init
git remote add -f origin https://github.com/XavierCHEN34/ClickSEG.git
git config core.sparseCheckout true
echo 'isegm' >> .git/info/sparse-checkout
git pull origin main
mv /tmp/clickseg "${FILE_DIR}"
cd "${FILE_DIR}"
cp setup.py clickseg/ && cd clickseg/ && pip install -e .
rm -rf /tmp/clickseg