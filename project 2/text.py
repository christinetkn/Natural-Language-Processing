#converts the text to its phonetics eg:
#This was easy for us. --> sil dh ih s w ao z iy z iy f r er ah s sil
#utterance_id_1 <space> <phonetics of text>

L1 = [] #store words
L2 = [] #store phonics
L = [] #store words of text

filesets = ['slp_lab2_data/filesets/validation_utterances.txt',
            'slp_lab2_data/filesets/test_utterances.txt',
            'slp_lab2_data/filesets/train_utterances.txt']
newfiles = ['dev/text', 'test/text','train/text']

file = open('slp_lab2_data/filesets/test_utterances.txt', 'r') #you will change this file three times
file1 = open('slp_lab2_data/transcription.txt', 'r')


for i in range(len(filesets)):

    file = open(filesets[i], 'r') #you will change this file three times
    file1 = open('slp_lab2_data/transcription.txt', 'r')
    lines = file.readlines()
    lines1 = file1.readlines()

    fp = open(newfiles[i], 'w') #you will change this file three times

    with open('slp_lab2_data/lexicon.txt','r') as file2:    
        for line in file2:
            count = 0
            phonics =''
            for word in line.split(" ", 1):
                if count==0:
                    removeSpecialChars = word.translate({ord(c): " " for c in "!@#$%^&*()[]{};/<>\|`~-=_+"})
                    removeSpecialChars = removeSpecialChars.translate({ord(c): "" for c in ".?,:"})
                    final = removeSpecialChars.lower()
                    L1.append(final)
                    count +=1
                else:
                    phonics = word

            L2.append(phonics)

    for line in lines:

        line = line.replace("\n", "")
        L = line.split("_")
        phon=''

        s = lines1[int(L[3])-1]

        right_string = s.split()
        for word in right_string:

            removeSpecialChars = word.translate ({ord(c): " " for c in "!@#$%^&*()[]{};/<>\|`~-=_+"})
            removeSpecialChars = removeSpecialChars.translate ({ord(c): "" for c in ".?,:"})

            final = removeSpecialChars.lower()
            if final == 'well kept':
                p = final.split()
                index1 = L1.index(p[0]+'\t')
                phon = phon+L2[index1]+' '
                index2 = L1.index(p[1]+'\t')
            else:
                index = L1.index(final+'\t')
                ph = L2[index][:-1]
                phon=phon+ph+' '

        final_phon = 'sil '+phon+'sil'
        fp.write(line + ' '+ final_phon +'\n')
    file1.close()
    file2.close()
    fp.close()
