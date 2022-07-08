def indexing():
  import string
  flag = True
  f = open("vocab/chars.syms", "w")
  counter = 1
  f.write("<eps> 0\n") #index 0 for <eps>
  alphas = list(string.ascii_letters[:26]) #index every letter with numbers from 1 to 26
  for chr in alphas:
      if flag: flag = False;
      else: f.write('\n')
      f.write(chr+' '+str(counter))
      counter = counter + 1
  f.close
#function is similar to modifier
indexing()
