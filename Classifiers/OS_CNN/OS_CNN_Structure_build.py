def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1): 
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer/(in_channel*sum(prime_list)))
    return out_channel_expect

# def generate_layer_parameter_list(start,end,paramenter_number_of_layer_list):
    
#     in_channel = 1 
#     prime_list = get_Prime_number_in_a_range(start, end)
    
#     layer_parameter_list = []
#     for paramenter_number_of_layer in paramenter_number_of_layer_list:
#         out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        
#         tuples_in_layer= []
#         for prime in prime_list:
#             tuples_in_layer.append((in_channel,out_channel,prime))
#         in_channel =  len(prime_list)*out_channel
        
#         layer_parameter_list.append(tuples_in_layer)
    
#     tuples_in_layer_last = []
#     first_out_channel = len(prime_list)*get_out_channel_number(paramenter_number_of_layer_list[0], 1, prime_list)
#     tuples_in_layer_last.append((in_channel,first_out_channel,1))
#     tuples_in_layer_last.append((in_channel,first_out_channel,2))
#     layer_parameter_list.append(tuples_in_layer_last)
#     return layer_parameter_list

def generate_layer_parameter_list(start,end,paramenter_number_of_layer_list, in_channel = 1):
    prime_list = get_Prime_number_in_a_range(start, end)
    if prime_list == []:
        print('start = ',start, 'which is larger than end = ', end)
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        
        tuples_in_layer= []
        for prime in prime_list:
            tuples_in_layer.append((in_channel,out_channel,prime))
        in_channel =  len(prime_list)*out_channel
        
        layer_parameter_list.append(tuples_in_layer)
    
    tuples_in_layer_last = []
    first_out_channel = len(prime_list)*get_out_channel_number(paramenter_number_of_layer_list[0], input_in_channel, prime_list)
    tuples_in_layer_last.append((in_channel,first_out_channel,start))
    tuples_in_layer_last.append((in_channel,first_out_channel,start+1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list