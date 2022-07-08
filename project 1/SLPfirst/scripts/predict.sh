#!/usr/bin/env bash

# Run spell corrector for an input word

# Usage:
#   bash scripts/predict.sh MY_SPELL_CHECKER tst
# Output:
#   test

# Command line args
SPELL_CHECKER_COMPILED=${1}
WORD=${2}


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

###
# Make sure these files exist
CHARSYMS=${CURRENT_DIRECTORY}/vocab/chars.syms
WORDSYMS=${CURRENT_DIRECTORY}/vocab/words.syms
###
touch fsts/correct.fst
# Make input fst for the misspelled word
python3 mkfstinput.py ${WORD} |
    # Compile and compose with the spell checker
    fstcompile --isymbols=${CHARSYMS} --osymbols=${WORDSYMS} vocab/${WORD}.txt fsts/${WORD}.fst |
    fstcompose - fsts/S.fst |
    # Get shortest path and sort arcs
    fstshortestpath | 
    fstrmepsilon  | 
    
    # print output fst using words.syms
    fstprint  |
    # Get destination word (corrected)
    cut -f4 |
    # Ignore epsilon outputs
    grep -v "<eps>"
    # Ignore accepting state line
    head -n -1 |
    # Remove trailing new line
    tr -d '\n'