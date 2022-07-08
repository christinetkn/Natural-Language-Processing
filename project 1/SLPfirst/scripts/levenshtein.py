import string
def lev_maker():
    state = 0
    f = open('fsts/L.txt', 'w')
    alphas = list(string.ascii_letters[:26]) #index every letter with numbers from 1 to 26
    for c in alphas:
        f.write(str(state)+' '+str(state)+' <eps> '+c+' 1\n') #from epsilon to everything else with cost 1
    for c in alphas:
        f.write(str(state)+' '+str(state)+' '+c+' <eps> '+' 1\n') #from char to epsilon with cost one
    for c1 in alphas:
        for c2 in alphas:
            if(c2 != c1):
                f.write(str(state)+' '+str(state)+' '+c1+' '+c2+' 1\n') #from a char to a different one with cost 1
            else:
                f.write(str(state)+' '+str(state)+' '+c1+' '+c2+' 0\n')#from a char to itself no cost

    f.write(str(state))
    f.close
lev_maker()
