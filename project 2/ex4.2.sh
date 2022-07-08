source path.sh

#exercise 4.2.2
build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/unigram_train.ilm.gz
build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/bigram_train.ilm.gz
#the next lines are not necessary but they are requested
build-lm.sh -i data/local/dict/lm_test.text -n 1 -o data/local/lm_tmp/unigram_test.ilm.gz
build-lm.sh -i data/local/dict/lm_test.text -n 2 -o data/local/lm_tmp/bigram_test.ilm.gz
build-lm.sh -i data/local/dict/lm_dev.text -n 1 -o data/local/lm_tmp/unigram_dev.ilm.gz
build-lm.sh -i data/local/dict/lm_dev.text -n 2 -o data/local/lm_tmp/bigram_dev.ilm.gz

source path.sh
#exercise 4.2.3
compile-lm data/local/lm_tmp/unigram_train.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_ug.arpa.gz
compile-lm data/local/lm_tmp/bigram_train.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_bg.arpa.gz

source path.sh
#exercise 4.2.4
utils/prepare_lang.sh data/local/dict '' data/local/lang data/lang


source path.sh
#exercise 4.2.5
sort data/train/wav.scp -o  data/train/wav.scp
sort data/train/text -o  data/train/text
sort data/dev/wav.scp -o  data/dev/wav.scp
sort data/dev/text -o  data/dev/text
sort data/test/wav.scp -o  data/test/wav.scp
sort data/test/text -o  data/test/text

source path.sh
#exercise 4.2.6
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

#question 1
source path.sh
compile-lm data/local/lm_tmp/unigram_train.ilm.gz -eval=data/local/dict/lm_dev.text
compile-lm data/local/lm_tmp/unigram_train.ilm.gz -eval=data/local/dict/lm_test.text
compile-lm data/local/lm_tmp/bigram_train.ilm.gz -eval=data/local/dict/lm_dev.text
compile-lm data/local/lm_tmp/bigram_train.ilm.gz -eval=data/local/dict/lm_test.text

source path.sh
#exercise 4.2.7
./timit_format_data.sh
