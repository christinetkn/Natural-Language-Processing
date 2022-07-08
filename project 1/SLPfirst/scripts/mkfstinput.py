"""
USAGE:
    python mkfstinput.py MY_WORD > my_word.fst
OR:
    python mkfstinput.py MY_WORD | fstcompile | fstcompose - MY_SPELLCHECKER.fst | ...
"""

import sys



def make_input_fst(word):
    """Create an fst that accepts a word letter by letter
    This can be composed with other FSTs, e.g. the spell
    checker to provide an "input" word
    """
    state = 0     #zero is the first state
    curword = ""
    f = open("vocab/"+word+".txt", "w") #creates a file in vocab directory with the name of the word we are going to make automaton fot
    for i in word:
        curword = curword+i #this will produce the output of each state
        f.write(str(state) +' '+ str(state+1)+' '+i + ' '+ i+'\n') #this will write it in the file
        state = state + 1 #create next state
    f.write(str(state))
    f.close()




if __name__ == "__main__":
    word = sys.argv[1]
    make_input_fst(word)