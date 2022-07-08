source path.sh


./steps/make_mfcc.sh data/train
./steps/make_mfcc.sh data/test
./steps/make_mfcc.sh data/dev

./steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc
./steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test mfcc
./steps/compute_cmvn_stats.sh data/dev exp/make_mfcc/dev mfcc
