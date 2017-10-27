import numpy as np

input_conv = 'movie_conversations.txt'
input_line = 'movie_lines.txt'
output = 'cornell_pair'

input_conv_fp = open(input_conv,'r')
input_line_fp = open(input_line,'r')
fp = open(output,'w')

conversation = input_conv_fp.readlines()
lines = input_line_fp.readlines()

for pairs in conversation:
    l = pairs.split('[')[1].split(']')[0]
    print l
    p = []
    for t in l.split('\''):
        if t.find('L') < 0: continue
        p.append(t)

    if len(p) < 2: continue
    for i in range(len(p)-1):
        for line in lines:
            if p[i] in line:
                a = line.split(' +++$+++ ')[4].replace('\n','')
                break
        for line in lines:
            if p[i+1] in line:
                b = line.split(' +++$+++ ')[4].replace('\n','')
	        break

        if len(a) < 4: continue
        if len(b) < 4: continue
        fp.write(a+' aaaaapairaaaaa '+b+'\n')

fp.close()
input_line_fp.close()
input_conv_fp.close()
