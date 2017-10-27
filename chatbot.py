import parser as Parser
import numpy as np
import pickle
import sys

sys.path.insert(0, 'src/')
import tensorNet

SEQ_LEN = 32

if len(sys.argv) < 3:
    print 'Usage: python chatbot.py [Text pickle] [TRT Model]'
    sys.exit(0)


with open(sys.argv[1]) as f:
    wmap, iwmap = pickle.load(f)
Parser.runtimeLoad(wmap, iwmap)
print '[ChatBot] load word map from '+sys.argv[1]


engine = tensorNet.createTrtFromUFF(sys.argv[2])
tensorNet.prepareBuffer(engine)
print '[ChatBot] create tensorrt engine from '+sys.argv[2]


while True:
    b = raw_input('\n\n\x1b[1;105;97m'+'Please write your question(q for quite):'+'\x1b[0m')
    if b=='q':
        print 'Bye Bye!!'
        break
    elif len(b)>0:
        raw = Parser.runtimeParser(b+' ', SEQ_LEN)
        question = []
        question.append([wmap[c] for c in raw])

        _input  = np.array(question[0], np.int32)
        _output = np.zeros([SEQ_LEN], np.int32)
        tensorNet.inference(engine, _input, _output)
        print 'Q: '+'\x1b[1;39;94m'+Parser.decodeData(question[0])+'\x1b[0m'
        print 'A: '+'\x1b[1;39;92m'+Parser.decodeData(_output.tolist())+'\x1b[0m'
