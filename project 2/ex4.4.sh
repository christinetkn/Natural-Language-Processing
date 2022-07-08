source path.sh
#exercise 4.4.1
./steps/train_mono.sh --nj 4 data/train data/lang exp/mono

source path.sh
#exercise 4.4.2
utils/mkgraph.sh data/lang_phones_ug exp/mono exp/mono/graph_unigram

utils/mkgraph.sh data/lang_phones_bg exp/mono exp/mono/graph_bigram

source path.sh
#exercise 4.4.3

#decode for bigram at dev & test respectively
steps/decode.sh --nj 4 exp/mono/graph_bigram data/dev exp/mono/decode_bigram_validation
steps/decode.sh --nj 4 exp/mono/graph_bigram data/test exp/mono/decode_bigram_test

#exercise 4.4.3 has already been answered in exercise 4.4.3

#monophone model
#decode for unigram at dev & test respectively
steps/decode.sh --nj 4 exp/mono/graph_unigram data/dev exp/mono/decode_unigram_validation
steps/decode.sh --nj 4 exp/mono/graph_unigram data/test exp/mono/decode_unigram_test

source path.sh 
#exercise 4.4.5
steps/align_si.sh data/train data/lang exp/mono exp/mono_align
steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_align exp/tri1

utils/mkgraph.sh data/lang_phones_ug exp/tri1 exp/tri1/graph_unigram
utils/mkgraph.sh data/lang_phones_bg exp/tri1 exp/tri1/graph_bigram

#triphone model
#decode for bigram at dev & test respectively
steps/decode.sh --nj 4 exp/tri1/graph_bigram data/dev exp/tri1/decode_bigram_validation
steps/decode.sh --nj 4 exp/tri1/graph_bigram data/test exp/tri1/decode_bigram_test

#decode for unigram at dev & test respectively
steps/decode.sh --nj 4 exp/tri1/graph_unigram data/dev exp/tri1/decode_unigram_validation
steps/decode.sh --nj 4 exp/tri1/graph_unigram data/test exp/tri1/decode_unigram_test


