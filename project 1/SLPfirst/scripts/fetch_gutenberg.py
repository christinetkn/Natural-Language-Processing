def dictionary_maker(text_doc):
  
  dict = {} #empty dictionary
  with open(text_doc, "r" ) as file:
    for line in file:
      for x in line.split(): #read words from file
        if (x in dict):      #if the word just read is in dictionary
          dict[x] = dict[x]+1 #increase its value by one
        else:                 
          dict.update({x:1})  #else add word with value 1 in dictionary
  s = {k: dict[k] for k in sorted(dict)} #sort the dictionary lexicographically
  return s



def dictionary_filter(dic):
  list = []
  for i in dic: #take an empty list 
    if(dic[i]<5):   #and fill it with all the keys with value less than 5
      list.append(i)
  while(list):
    x = list.pop(0) #then delete every element of the dictionary that exists in the list
    dic.pop(x)
  return dic #filtering is done



def modifier(dictionary):
  f = open("vocab/words.vocab.txt", "w")
  flag = True
  for key, value in dictionary.items():
     if flag: flag = False;   #first time only write
     else: f.write('\n')    #for every other time before writing add new line
     f.write(key+' '+str(value))    #that prevents having a new line at the end of the file
  f.close

mydict = dictionary_maker("gutenberg.txt") #make a dictionary for file gutenberg.txt
filtered = dictionary_filter(mydict) #remove all keys with value less than 5
modifier(filtered)

  
