scripts/automaton_maker.py          συναρτήσεις για την παραγωγή αυτομάτου από ένα αρχείο με λέξεις, φτιάχνει το αυτόματο του V
scripts/corpus_script.txt           δοσμένο script για δημιουργία corpus
scripts/dict_index.py               κάνει indexing σε λεξικό που παράγεται με το fetch_gutenberg.py
scripts/gutenberg.txt               περιέχει το corpus
scripts/index.py                    κάνει index τα γράμματα a-z και <eps>
scripts/levenshtein.py              δημιουργεί μετατροπέα L βασισμένο σε απόσταση levenshtein
scripts/levfst			    shell script που δημιουργεί το L.pdf
scripts/mkfstinput.py		    φτιάχνει αυτόματο για μια λέξη
scripts/predict.sh		    shell script που παίρνει μια λάθος λέξη και μέσω του spell checker δημιουργεί τη σωστή
scripts/spell_checker	            shell script που δημιουργεί τον spell checker
scripts/spell_test.txt		    έχει λέξεις με πιθανά λάθη

#################################################################################################################################

vocab/chars.syms		   παράγει τα σύμβολα εισόδου με χρήση της scripts/index.py
vocab/lexicon_subset.txt	   υποσύνολο του συνολικού λεξικού  
vocab/words.syms		   περιέχει τις λέξεις του λεξικού με indexing με χρήση της scripts/dict_index.py
vocab/words.vocab.txt		   το λεξικό με πλήθος εμφάνισης κάθε λέξης, έχει γίνει φιτράρισμα σε όσες εμφανίζονται <5 φορές

#################################################################################################################################

fsts/det.pdf			  ντετερμινιστικό αυτόματο για το 5β
fsts/eps.pdf			  το αυτόματο του 5β χωρίς τις μεταβάσεις "ε"			  
fsts/L.pdf			  Levenshtein tranduser
fsts/L1.pdf			  smaller Levenshtein tranducer
fsts/minimum.pdf		  το αυτόματο του 5β minimized
fsts/preprocessed.pdf		  το αυτόματο του 5β χωρίς κάποια επεξεργασία
fsts/SC.pdf			  spell checker
fsts/SC1.pdf			  υποσύνολο spell checker	
fsts/SubV.pdf			  αυτόματο με υποσύνολο του V




