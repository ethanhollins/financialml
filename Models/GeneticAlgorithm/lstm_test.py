import bt
import numpy as np

np.random.seed(1)

def forget_gate(x, h, Weights_hf, Bias_hf, Weights_xf, Bias_xf, prev_cell_state):
	forget_hidden = np.dot(Weights_hf, h) + Bias_hf
	print(Weights_xf.shape)
	print(x.shape)
	print(Bias_xf.shape)
	forget_eventx = np.dot(Weights_xf, x).T + Bias_xf
	return np.multiply(bt.sigmoid(forget_hidden + forget_eventx), prev_cell_state)

def input_gate(
	x, h, 
	Weights_hi, Bias_hi, 
	Weights_xi, Bias_xi, 
	Weights_hl, Bias_hl, 
	Weights_xl, Bias_xl
):
	ignore_hidden = np.dot(Weights_hi, h) + Bias_hi
	ignore_eventx = np.dot(Weights_xi, x).T + Bias_xi
	learn_hidden = np.dot(Weights_hl, h) + Bias_hl
	learn_eventx = np.dot(Weights_xl, x).T + Bias_xl
	return np.multiply(bt.sigmoid(ignore_eventx + ignore_hidden), np.tanh(learn_eventx + learn_hidden))

def cell_state(forget_gate_output, input_gate_output):
	return forget_gate_output + input_gate_output

def output_gate(x, h, Weights_ho, Bias_ho, Weights_xo, Bias_xo, cell_state):
	out_hidden = np.dot(Weights_ho, h) + Bias_ho
	out_eventx = np.dot(Weights_xo, x).T + Bias_xo
	return np.multiply(bt.sigmoid(out_eventx + out_hidden), np.tanh(cell_state))

def model_output(lstm_output, fc_Weight, fc_Bias):
  '''Takes the LSTM output and transforms it to our desired 
  output size using a final, fully connected layer'''
  # print(lstm_output.shape)
  # print(fc_Weight.shape)
  return np.dot(lstm_output, fc_Weight) + fc_Bias

# Set Parameters for a small LSTM network
input_size = 2 # size of one 'event', or sample in our batch of data
hidden_dim = 3 # 3 cells in the LSTM layer
output_size = 1 # desired model output

# Initialize Weights and Biases
Weights_xi = np.random.normal(size=(hidden_dim, input_size))
Weights_xf = np.random.normal(size=(hidden_dim, input_size))
Weights_xl = np.random.normal(size=(hidden_dim, input_size))
Weights_xo = np.random.normal(size=(hidden_dim, input_size))

Bias_xi = np.random.normal(size=(hidden_dim))
Bias_xf = np.random.normal(size=(hidden_dim))
Bias_xl = np.random.normal(size=(hidden_dim))
Bias_xo = np.random.normal(size=(hidden_dim))

Weights_hi = np.random.normal(size=(hidden_dim, hidden_dim))
Weights_hf = np.random.normal(size=(hidden_dim, hidden_dim))
Weights_hl = np.random.normal(size=(hidden_dim, hidden_dim))
Weights_ho = np.random.normal(size=(hidden_dim, hidden_dim))

Bias_hi = np.random.normal(size=(hidden_dim))
Bias_hf = np.random.normal(size=(hidden_dim))
Bias_hl = np.random.normal(size=(hidden_dim))
Bias_ho = np.random.normal(size=(hidden_dim))

fc_Weight = np.random.normal(size=(hidden_dim, output_size))
fc_Bias = np.random.normal(size=(output_size))


# Simple Time Series Data
data = np.array(
	[[1,1],
	 [2,2],
	 [3,3],
	 [4,4]]
)

# Initialize cell and hidden states with zeroes
h = np.zeros(hidden_dim)
c = np.zeros(hidden_dim)
print(data.shape)
print(data.T.shape)
# Loop through data, updating the hidden and cell states after each pass
# for eventx in data:
f = forget_gate(data.T, h, Weights_hf, Bias_hf, Weights_xf, Bias_xf, c)
i = input_gate(data.T, h, Weights_hi, Bias_hi, Weights_xi, Bias_xi,
		Weights_hl, Bias_hl, Weights_xl, Bias_xl)
c = cell_state(f, i)
h = output_gate(data.T, h, Weights_ho, Bias_ho, Weights_xo, Bias_xo, c)
print(model_output(h, fc_Weight, fc_Bias))
