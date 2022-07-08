#creates wav.scp files: utterance_id_1 <space> /path/to/wav1
filesets = ['slp_lab2_data/filesets/validation_utterances.txt',
            'slp_lab2_data/filesets/test_utterances.txt',
            'slp_lab2_data/filesets/train_utterances.txt']
newfiles = ['dev/wav.scp', 'test/wav.scp','train/wav.scp']

for i in range(len(filesets)):
    file = open(filesets[i], 'r')
    lines = file.readlines()

    fp = open(newfiles[i], 'w') 

    for line in lines:
        line = line.replace("\n", "")
        L = line.split("_")
        fp.write(line+' '+"data/slp_lab2_data/wav/"+L[2]+"/"+line+".wav"+"\n")
    fp.close()
