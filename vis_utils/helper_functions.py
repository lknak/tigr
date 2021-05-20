# Helper function to print out relation between losses and network parameters
# loss_list given as: [(name, loss_variable), ...]
# named_parameters_list using pytorch function named_parameters(): [(name, network.named_parameters()), ...]
def print_loss_params_relation(loss_list, named_parameters_list):
    loss_variables = {}
    for name, loss in loss_list:
        if loss.grad_fn is None:
            variables_ = []
        else:
            def recursive_sub(loss):
                r = []
                if hasattr(loss, 'next_functions'):
                    for el, _ in loss.next_functions:
                        if el is None:
                            continue
                        if hasattr(el, 'variable'):
                            r.append(el.variable)
                        else:
                            r += recursive_sub(el)
                return r

            variables_ = recursive_sub(loss=loss.grad_fn)
        loss_variables[name] = variables_

    # Assign params to networks and check for affection
    max_char_length = 0
    affected_network_params = {}
    for network_name, named_parameters in named_parameters_list:
        affected_params = {}
        for n, p in named_parameters:
            # Ignore bias since this will just duplicate the outcome
            if (p.requires_grad) and ('bias' not in n):
                for loss_name in loss_variables.keys():
                    # Skip if loss_name is already existing in affected params
                    if n in affected_params.keys() and loss_name in affected_params[n]:
                        continue
                    # Iterate through all variables
                    for v in loss_variables[loss_name]:
                        if v.shape == p.shape and (v.data == p.data).all():
                            if n in affected_params.keys():
                                affected_params[n].append(loss_name)
                            else:
                                affected_params[n] = [loss_name]
                                max_char_length = len(n) if len(n) > max_char_length else max_char_length
                            # Exit for after assigning loss name to param
                            break
        affected_network_params[network_name] = affected_params

    # Print out
    for network_name in affected_network_params.keys():
        print(f'Affected Params for {network_name}:')
        if len(affected_network_params[network_name].keys()) == 0:
            print('\t-')
        else:
            for p_name in affected_network_params[network_name].keys():
                print(f'\t{p_name}:{" " * (max_char_length - len(p_name))}\t{affected_network_params[network_name][p_name]}')
        print('')