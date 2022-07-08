fp1 = open('local/dict/silence_phones.txt', 'w')
fp2 = open('local/dict/optional_silence.txt', 'w')
fp3 = open('local/dict/lexicon.txt', 'w')
fp4 = open('local/dict/nonsilence_phones.txt', 'w')
fp5 = open('local/dict/extra_questions.txt', 'w')

fp1.write('sil'+'\n')
fp2.write('sil'+'\n')

phoenemes = ['<oov>','aa', 'ae', 'ah', 'ao', 'aw','ay', 'b', 'ch', 'd', 'dh','eh','er','ey','f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r','s','sh','sil','t','th','uh','uw','v','w','y','z','zh']

phoenemes.sort()

for phoeneme in phoenemes:
    if phoeneme != 'sil':
        fp4.write(phoeneme+'\n')
    fp3.write(phoeneme+' '+phoeneme+'\n')
fp5.write('')

fp1.close()
fp2.close()
fp3.close()
fp4.close()
fp5.close()

filesets = ['dev/text', 'test/text','train/text']
newfiles = ['local/dict/lm_dev.text', 'local/dict/lm_test.text', 'local/dict/lm_train.text']

for i in range(len(filesets)):
    file = open(filesets[i], 'r')
    fp6 = open(newfiles[i], 'w')

    for line in file:
        line.replace("\n", "")
        L = line.split(' ')
        L[1] = '<s>'
        L[len(L)-1] = '</s>'
        L.pop(0) #addition
        st = ''
        for word in L:
            st=st+word+' '
        st = st[:-1]
        fp6.write(st+"\n")
    fp6.close()
    file.close()
