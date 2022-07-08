#CREATES UTT2SPK FILES: utterance_id_1 <space> speaker_id
filesets = ['slp_lab2_data/filesets/validation_utterances.txt',
            'slp_lab2_data/filesets/test_utterances.txt',
            'slp_lab2_data/filesets/train_utterances.txt']
newfiles = ['dev/utt2spk', 'test/utt2spk','train/utt2spk']

for i in range(len(filesets)):
    file = open(filesets[i], 'r')
    lines = file.readlines()

    fp = open(newfiles[i], 'w') 

    for line in lines:
        line = line.replace("\n", "")
        L = line.split("_")
        fp.write(line +" "+L[2]+"\n")

    fp.close()