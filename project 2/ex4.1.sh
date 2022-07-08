cp ../wsj/s5/cmd.sh cmd.sh
cp ../wsj/s5/path.sh path.sh

ln -s ../wsj/s5/steps
ln -s ../wsj/s5/utils


cd data
mkdir local
cd local
ln -s ../../steps/score_kaldi.sh

cd ../..
mkdir conf

cd data 
mkdir lang

cd local
mkdir dict 
mkdir lm_tmp
mkdir nist_lm
