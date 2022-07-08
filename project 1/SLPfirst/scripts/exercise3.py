#viterbi exercise
transmitions = [[0.25, 0.2, 0.3, 0.25], [0.2, 0.25, 0.3, 0.25],[0.4, 0.2, 0.2, 0.2],[0.25, 0.3, 0.2, 0.25]]
p = 0.25
observations = ['U','V','U','V','V','V','U','U','V','U']
pV = [0.5, 0.8, 0.25, 0.2]
pU = [0.5, 0.2, 0.75, 0.8]

#initialize di values:
d = [p*0.5, p*0.2, p*0.75, p*0.8]

#next step: find dt(j) = max[dt-1(i)*aij]*bj(Ot)
#find which state has the maximum d:

i = d.index(max(d)) 
max_dt = d[i] 
print('max is:d1(2) = '+str(max_dt))
print('d1 array is: ', d, '\n')
path = [i+1]

for k in range(1, len(observations)):
    if observations[k] == 'U':
        d = []
        for j in range(4):
            d.append(max_dt*transmitions[i][j]*pU[j])
        i = d.index(max(d))
        max_dt = d[i]
        print('max is:'+'d'+str(k+1)+'('+str(i+1)+')'+' = '+ str(max_dt))
        print('d'+str(k+1)+' array is: ', d, '\n')
        path.append(i+1)
    else:
        d = []
        for j in range(4):
            d.append(max_dt*transmitions[i][j]*pV[j])
        i = d.index(max(d))
        max_dt = d[i]
        print('max is:'+'d'+str(k+1)+'('+str(i+1)+')'+' = '+ str(max_dt))
        print('d'+str(k+1)+' array is: ', d, '\n')
        path.append(i+1)

print('The most likely path with the Viterbi algorithm is:', path,'\n' )
print('P(O,Q|l) = ', max_dt)
