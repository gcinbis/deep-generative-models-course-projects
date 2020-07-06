cd $(dirname $0)
mkdir -p checkpoint/wae-mmd/
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1w1FKCekcz-OlVozmjRIDATrj4zpzf8Nt' -O checkpoint/wae-mmd/encoder_55.pt  
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vOP_2rh2RSRt_TViQ6cXp85GceGhXd--' -O checkpoint/wae-mmd/decoder_55.pt 