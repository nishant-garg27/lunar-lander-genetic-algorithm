import numpy 
def policy_action(flat_params, input_obs):
    first_bias = flat_params[128:144]
    second_bias = flat_params[208:]
    first_weights = flat_params[:128].reshape(8, 16)
    second_weights = flat_params[144:208].reshape(16, 4)
    hidden_layer = input_obs @ first_weights
    hidden_layer = hidden_layer + first_bias
    activated_hidden = numpy.maximum(hidden_layer, 0)
    raw_output = activated_hidden @ second_weights
    raw_output = raw_output + second_bias
    action_index = numpy.argmax(raw_output)
    return int(action_index)