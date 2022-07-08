def dict_indexing(diction):
  counter = 0
  flag = True
  destfile= open("vocab/words.syms", "w") #create a destination file
  with open(diction, "r" ) as file:   #open and read the dictionary
   for line in file:
     for word in line.split():      # for every word in dictionary
       if flag: flag = False;
       else: destfile.write('\n')
       if (type(word) == str): #if it is a word and not an int
         destfile.write(word+' '+str(counter))  #put it in destfile with an index
         counter = counter+1
         break
   destfile.close()
dict_indexing("vocab/words.vocab.txt")