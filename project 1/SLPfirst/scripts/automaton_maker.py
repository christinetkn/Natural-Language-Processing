import os  


#this creates the txt file we will compile and make automata 
def make_automaton(s):
  state = 0     #zero is the first state
  curword = ""
  f = open("vocabs/"+s+".txt", "w") #creates a file in vocab directory with the name of the word we are going to make automaton fot
  for i in s:
      curword = curword+i #this will produce the output of each state
      f.write(str(state) +' '+ str(state+1)+' '+i + ' '+ i+'\n') #this will write it in the file
      state = state + 1 #create next state
  f.write(str(state))
  f.close()

##ACCEPTS EVERY WORD IN DICTIONARY
##IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!
##change vocab/lexicon_subset.txt with vocab/words.vocab.txt
def make_automata():
  flag = True   #goes false when reading first word
  with open('vocab/words.vocab.txt', 'r') as file:
      for line in file:
        for word in line.split():
          if flag:        #case for first word
            make_automaton(word)  #make the file for automaton
            os.system("fstcompile --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms --keep_isymbols --keep_osymbols vocabs/" + word+ ".txt fstss/"  +word+".fst") #make it .fst
            os.system('cp fstss/'  +word+".fst fsts/V.fst") #and put it in V.fst
            flag = False; #turn flag false to read other words
            break
          else:
              make_automaton(word)
              os.system("fstcompile --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms --keep_isymbols --keep_osymbols vocabs/" + word+ ".txt fstss/"  +word+".fst") #compile
              os.system('fstunion fsts/V.fst fstss/'+word+'.fst fsts/V.fst') #fstunion of each word in the vocab
              break
      os.system('rm -rf fstss')
      os.system('rm -rf vocabs')
      file.close()

def minimal_automata():
  make_automata()
  os.system('fstrmepsilon fsts/V.fst fsts/V.fst') #cuts epsilon transitions
  os.system('fstdeterminize fsts/V.fst fsts/V.fst') #makes automaton deterministic
  os.system('fstminimize fsts/V.fst fsts/V.fst')  #minimizes automaton
  os.system('fstdraw fsts/V.fst >fsts/minimum.dot')
  os.system('dot -Tpdf -ofsts/minimum.pdf fsts/minimum.dot')
  os.system('xdg-open fsts/minimum.pdf') #open minimum automaton created above

minimal_automata()
