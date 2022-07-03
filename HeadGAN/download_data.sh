wget https://github.com/cleardusk/3DDFA_V2/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
mv 3DDFA_V2-master TDDFA_V2
cd TDDFA_V2/
./build.sh
cd ..
gdown https://drive.google.com/uc?id=1S9wB6iS5ZyTAyp5O56swQqn59meCTlHd
unzip TDDFA_V2_modifications.zip
rm TDDFA_V2_modifications.zip
cp -rf TDDFA_V2_modifications/* TDDFA_V2
rm -rf TDDFA_V2_modifications
mkdir data
mkdir data/test/
mkdir data/test/aac
mkdir data/test/mp4
wget --user USERNAME --password PASSWORD -P data/test/aac http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_test_aac.zip
wget --user USERNAME --password PASSWORD -P data/test/mp4 http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_test_mp4.zip
cd data/test/aac
unzip vox2_test_aac.zip
rm vox2_test_aac.zip
cd ../mp4
unzip vox2_test_mp4.zip
rm vox2_test_mp4.zip
