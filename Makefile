
# DIRNAME=$(basename $0)
DIR_NAME:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# suggest python 3.9/3.10
TORCHVERSION="1.12.1"
TORCHVISIONVERSION="0.13.1"
TORCHAUDIOVERSION="0.12.1"

.PHONY: install
install:
	@pip install torch==$(TORCHVERSION) \
		torchvision==$(TORCHVISIONVERSION) \
		torchaudio==$(TORCHAUDIOVERSION) \
		-f https://download.pytorch.org/whl/torch_stable.html 
	@pip install openmim
	@mim install mmcv
	@pip install -r requirements.txt
	@sh ${DIR_NAME}/install_clickseg.sh