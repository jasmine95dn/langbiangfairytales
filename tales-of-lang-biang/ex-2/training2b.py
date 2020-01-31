import tensorflow as tf
import numpy as np
from sys import argv
from subprocess import call
import matplotlib.pyplot as plt
'''
Data processing
'''
# read in data
fullstr = ""
with open("file.txt", "r") as opf:
    fullstr = "".join(opf.readlines())

'''
Vocabulary generation
'''
# define the vocabulary (=characters)
# include the start-of-sequence symbol <s>
# define a mapping from indices to characters 
# use an iteration over the training data instead of set() to make generation deterministic
char2id = {"<s>": 0}
for c in fullstr:
    if c not in char2id.keys():
        char2id[c] = len(char2id)
        
# vocabulary to generate new text
id2char = {i:c for (c, i) in char2id.items()}

'''
LSTM-RNN Model 
'''

# define network hyperparameters
sequence_length = 100 
vocab_size = len(char2id)
emb_size = 40
num_units = 200

# training parameters
batch_size = 16
iterations = 50000
print_save_freq = 500

# define the model
class LSTMCharRNN():
    def __init__(self, 
                 sequence_length=sequence_length, 
                 vocab_size=vocab_size, 
                 emb_size=emb_size, 
                 num_units=num_units, 
                 is_training=False, 
                 sampling=True,
                 start_id=char2id["\n"], 
                 temperature=1.0):
        
        # define placeholders
        self.x = tf.placeholder(shape=[None, sequence_length], name="x", dtype=tf.int32)
        self.y = tf.placeholder(shape=[None, sequence_length], name="y", dtype=tf.int32)
        
        # define variables
        emb = tf.get_variable(shape=[vocab_size, emb_size], name="emb")
        out_W = tf.get_variable(shape=[num_units, vocab_size], name="W_out")
        out_b = tf.get_variable(shape=[vocab_size], name="b_out", initializer=tf.zeros_initializer)
        
        # define 2-layer LSTM cell 
        # link of Multilayer-LSTM cell
        
        cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, activation=tf.tanh)
        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=num_units, activation=tf.tanh)
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell2])
        
        # define initial state for RNN 
        if is_training:
            batch_size = tf.shape(self.x)[0]
        else:
            # during inference, we work with batches of size 1 (for simplification)
            batch_size = 1
            
        zero_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        previous_state = zero_state
        
        # during inference, the x is not given. 
        # Instead we start from a start symbol and embed it.
        if not is_training:
            previous_output = tf.expand_dims(
                tf.nn.embedding_lookup(emb, tf.constant(start_id)), axis=0)
            
        
        # embed inputs
        embedded_inputs = tf.nn.embedding_lookup(emb, self.x)  # batch x seq_len x embedding_size 
        
        # RNN loop over embedded inputs
        total_loss = []
        outputs = []
        hidden_states = []
        for i in range(sequence_length):
            if is_training:
                # during training, the full input sequence is known
                current_input = embedded_inputs[:, i, :]   # batch x embedding_size
            else:
                # during inference, we generate the sequence with the model itself
                # inputs to the next step are the previous predictions
                current_input = previous_output
                
            current_y = self.y[:, i]                 
            
            # R
            new_output, new_state = rnn_cell(current_input, previous_state)
            previous_state = new_state
            hidden_states.append(new_output)
            
            # O
            current_output = tf.matmul(new_output, out_W) + out_b
            
            # cross-entropy loss
            current_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=current_y, logits=current_output)
            # collect losses over the full sequence
            total_loss.append(current_loss)
            
            # compute the prediction
            if not sampling:
                # take the most likely of output classes (=characters)
                prediction = tf.argmax(current_output, axis=1)
            else:
                # sample from the distribution over output classes (=characters)
                # squeezing is necessary because we sample only once
                prediction = tf.squeeze(tf.multinomial(current_output/temperature, num_samples=1), axis=1)
            # collect predictions over full sequence
            outputs.append(prediction)
    
            if not is_training:
                # if the full input sequence is not known (during inference), 
                # we treat the current prediction as input for the next step
                previous_output = tf.nn.embedding_lookup(emb, prediction)
        
        # sum the cross-entropy over time steps, average it over the batch
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.stack(total_loss, axis=1), axis=1),axis=0) 
        
        # stack the predictions from each time step into one tensor
        self.outputs_stacked = tf.stack(outputs, axis=1)
        
        # stack the hidden states, 2b
        self.new_output_stacked = tf.stack(hidden_states, axis=1)
        
        
        
        # optimizer: Adam
        opt = tf.train.AdamOptimizer()
        self.update_op = opt.minimize(self.loss)

model_dir = "Ep_1"

# create the model directory if it doesn't exist
# call('mkdir '+model_dir, shell=True)

# number of checkpoints to keep
max_to_keep = 1
load_model_dir = model_dir

tf.reset_default_graph()
g = tf.Graph()

with g.as_default():
	if argv[1] == 'training':
		# create the model
		m = LSTMCharRNN(sequence_length=111, vocab_size=vocab_size, emb_size=emb_size, num_units=num_units, is_training=True, sampling=False)
	    
	    # create a saver to reload model parameters
		saver = tf.train.Saver()
	    
		# start a session
		sess = tf.Session(graph=g)
		
		# restore the model parameters
		saver.restore(sess, "{}/model-{}".format(load_model_dir, 39000))
		print("Model restored from iteration {}.\n".format(39000))
		
		inputs_ = ['Mà việc con so sánh thầy Haifai với trùm Hắc Ám là sai lầm lớn nhất trong những sai lầm mà con có thể mắc phải.', '- Ờ, đó là cái dở của vợ chồng thầy Haifai... - Thầy N’Trang Long gật đầu một cách miễn cưỡng. - Ta nói hoài mà', 'Sáng nay, cả Nguyên lẫn Kăply đều muốn bịnh khi vừa mở mắt ra đã thấy con số 3 nằm đe dọa ngay trên trần nhà. Y', 'cả chục cái đầu ngoảnh tới ngoảnh lui nhặng xị, cố tìm xem con basilic đang ở đâu, quên phắt rằng đó là cách tố', ' sẽ trở thành người thống trị tối cao ở xứ Lang Biang này. Tương lai của trò sẽ rạng rỡ hơn bất cứ Đại phù thủy', '- Thưa cô... thưa cô... em mơ thấy... một con basilic... \n- Một con basilic! - Thầy Haifai la hoảng. - Làm sao ']
		
		inputs__ = [[i for i in input_] for input_ in inputs_]
		a = 1
		for i_ in inputs__:
			
			# convert the words to indices
			sample_input_indices = np.expand_dims([char2id[c] for c in i_], axis=0)
            
			# convert to arrays
			batch_inputs = np.array(sample_input_indices, dtype=int)
			
		
			# feed inputs to the network, retrieve loss and predictions
			hidden_states_stacked  = sess.run([m.new_output_stacked],
							     feed_dict={m.x: batch_inputs})
		
			# slicing for 5 neurons
			#print(hidden_states_stacked[0].shape)
			
			plt.matshow(hidden_states_stacked[0][0,:,[1,16,30,175,117]])
			plt.xticks(np.arange(len(i_)), i_)
			plt.yticks(np.arange(5), [1,16,30,175,117])
			plt.colorbar()
			plt.savefig('training2b_'+str(a)+'.png')
			plt.show()
			a +=1
			
        








        
