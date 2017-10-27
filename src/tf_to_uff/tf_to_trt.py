import tensorrt as trt
import model
import sys
import uff

if len(sys.argv) < 3:
    print 'Usage: python tf_to_trt.py [TF Model] [TRT Model]'
    sys.exit(0)
else:
    print '[ChatBot] Covert '+sys.argv[1]+' to '+sys.argv[2]

tf_model = model.getChatBotModel(sys.argv[1])
uff_model = uff.from_tensorflow(tf_model, ['h0_out','c0_out','h1_out','c1_out','final_output'], output_filename=sys.argv[2], text=False)
print '[ChatBot] Successfully transfer to UFF model'
