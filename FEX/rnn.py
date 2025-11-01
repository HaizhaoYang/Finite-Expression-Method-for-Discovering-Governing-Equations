###############################################################################
# General Information
###############################################################################
# Author: ZhongYi Jiang

# rnn.py: Houses the RNN model used to sample expressions. Supports batched
# sampling of variable length sequences. Can select RNN, LSTM, or GRU models.

###############################################################################
# Dependencies
###############################################################################


import torch.nn as nn
import torch.nn.functional as F
import torch
import time

###############################################################################
# Sequence RNN Class
###############################################################################

class DSRRNN(nn.Module):
    def __init__(self, operators, hidden_size, device, min_length=2, max_length=15, type='rnn', num_layers=1, dropout=0.0, apply_constraint=True):
        super(DSRRNN, self).__init__()

        self.input_size = 2*len(operators) # One-hot encoded parent and sibling
        self.hidden_size = hidden_size
        self.output_size = len(operators) # Output is a softmax distribution over all operators
        self.num_layers = num_layers
        self.dropout = dropout
        self.operators = operators
        self.device = device
        self.type = type
        self.apply_constraint = apply_constraint

        # Initial cell optimization
        self.init_input = nn.Parameter(data=torch.rand(1, self.input_size), requires_grad=True).to(self.device)
        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, self.hidden_size), requires_grad=True).to(self.device)

        self.min_length = min_length
        self.max_length = max_length

        if (self.type == 'rnn'):
            self.rnn = nn.RNN(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                dropout = self.dropout
            ).to(self.device)
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        elif (self.type == 'lstm'):
            self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                proj_size = self.output_size,
                dropout = self.dropout
            ).to(self.device)
            self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, self.output_size), requires_grad=True).to(self.device)
        elif (self.type == 'gru'):
            self.gru = nn.GRU(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first = True,
                dropout = self.dropout
            ).to(self.device)
            self.projection_layer = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.activation = nn.Softmax(dim=1)

    def sample_sequence(self, n, min_length=2, max_length=15,apply_constraint =True):
        '''
             Algorithm 2 Sampling an expression from the RNN
        '''
        sequences = torch.zeros((n, 0))
        entropies = torch.zeros((n, 0)) # Entropy for each sequence
        log_probs = torch.zeros((n, 0)) # Log probability for each token

        sequence_mask = torch.ones((n, 1), dtype=torch.bool)

        input_tensor = self.init_input.repeat(n, 1)
        hidden_tensor = self.init_hidden.repeat(n, 1)
        if (self.type == 'lstm'):
            hidden_lstm = self.init_hidden_lstm.repeat(n, 1)

        counters = torch.ones(n) # Number of tokens that must be sampled to complete expression
        lengths = torch.zeros(n) # Number of tokens currently in expression

        # While there are still tokens left for sequences in the batch
        trig_indicate = torch.zeros((n), dtype=torch.bool)
        parent_sibling = None
        while(sequence_mask.all(dim=1).any()):
            if (self.type == 'rnn'):
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)
            elif (self.type == 'lstm'):
                output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm)
            elif (self.type == 'gru'):
                output, hidden_tensor = self.forward(input_tensor, hidden_tensor)


            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, lengths, sequences, trig_indicate, parent_sibling, apply_constraint)
            output = output / torch.sum(output, axis=1)[:, None]

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()



            # Add sampled tokens to sequences
            sequences = torch.cat((sequences, token[:, None]), axis=1)
            lengths += 1

            # Add log probability of current token
            log_probs = torch.cat((log_probs, dist.log_prob(token)[:, None]), axis=1)

            # Add entropy of current token
            entropies = torch.cat((entropies, dist.entropy()[:, None]), axis=1)

            # Update counter
            counters -= 1
            counters += torch.isin(token, self.operators.arity_two).long() * 2
            counters += torch.isin(token, self.operators.arity_one).long() * 1

            # Update sequence mask
            # This is for the next token that we sample. Basically, we know if the
            # next token will be valid or not based on whether we've just completed the sequence (or have in the past)
            sequence_mask = torch.cat(
                (sequence_mask, torch.bitwise_and((counters > 0)[:, None], sequence_mask.all(dim=1)[:, None])),
                axis=1
            )

            # Compute next parent and sibling; assemble next input tensor
            parent_sibling = self.get_parent_sibling(sequences, lengths)
            input_tensor = self.get_next_input(parent_sibling)

        # Filter entropies log probabilities using the sequence_mask
        entropies = torch.sum(entropies * (sequence_mask[:, :-1]).long(), axis=1)
        log_probs = torch.sum(log_probs * (sequence_mask[:, :-1]).long(), axis=1)
        sequence_lengths = torch.sum(sequence_mask.long(), axis=1)
        return sequences, sequence_lengths, entropies, log_probs

    def forward(self, input, hidden, hidden_lstm=None):
        """Input should be [parent, sibling]
        """
        if (self.type == 'rnn'):
            output, hidden = self.rnn(input[:, None].float(), hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hidden[0, :]
        elif (self.type == 'lstm'):
            output, (hn, cn) = self.lstm(input[:, None].float(), (hidden_lstm[None, :], hidden[None, :]))
            output = self.activation(output[:, 0, :])
            return output, cn[0, :], hn[0, :]
        elif (self.type == 'gru'):
            output, hn = self.gru(input[:, None].float(), hidden[None, :])
            output = output[:, 0, :]
            output = self.projection_layer(output)
            output = self.activation(output)
            return output, hn[0, :]

    def apply_constraints(self, output, counters, lengths, sequences, trig_indicate, parent_sibling = None, apply_constraint = True):
        """ Subroutine 2 Applying generic constraints in situ when sampling from the RNN
            Applies in situ constraints to the distribution contained in output based on the current tokens
        """
        # Add small epsilon to output so that there is a probability of selecting
        # everything. Otherwise, constraints may make the only operators ones
        # that were initially set to zero, which will prevent us selecting
        # anything, resulting in an error being thrown
        epsilon = torch.ones(output.shape) * 1e-20
        output = output + epsilon.to(self.device)



        # ~ Check that minimum length will be met ~
        # Explanation here
        min_boolean_mask = (counters + lengths >= torch.ones(counters.shape) * self.min_length).long()[:, None]
        min_length_mask = torch.max(self.operators.nonzero_arity_mask[None, :], min_boolean_mask)
        output = torch.minimum(output, min_length_mask)


        # ~ Check that maximum length won't be exceed ~
        max_boolean_mask = (counters + lengths <= torch.ones(counters.shape) * (self.max_length - 2)).long()[:, None]
        max_length_mask = torch.max(self.operators.zero_arity_mask[None, :], max_boolean_mask)
        output = torch.minimum(output, max_length_mask)

        # ~ Ensure that all expressions have a variable ~
        nonvar_zeroarity_mask = (~torch.logical_and(self.operators.zero_arity_mask, self.operators.nonvariable_mask)).long()
        if (lengths[0].item() == 0.0): # First thing we sample can't be
            output = torch.minimum(output, nonvar_zeroarity_mask)
        else:
            nonvar_zeroarity_mask = nonvar_zeroarity_mask.repeat(counters.shape[0], 1)
            # Don't sample a nonvar zeroarity token if the counter is at 1 and
            # we haven't sampled a variable yet
            counter_mask = (counters == 1)
            contains_novar_mask = ~(torch.isin(sequences, self.operators.variable_tensor).any(axis=1))
            last_token_and_no_var_mask = (~torch.logical_and(counter_mask, contains_novar_mask)[:, None]).long()
            nonvar_zeroarity_mask = torch.max(nonvar_zeroarity_mask, last_token_and_no_var_mask * torch.ones(nonvar_zeroarity_mask.shape)).long()
            output = torch.minimum(output, nonvar_zeroarity_mask)

        if apply_constraint:
            # ~ Ensure the child of a unary operator should not be the inverse of that operator ~
            if (lengths[0].item() != 0.0):
                unary_constrain_boolean = (sequences[:, sequences.shape[1] - 1][:, None] == self.operators.unaryvaribale_tensor).any(dim=1)
                unary_constrain_mask = torch.ones((output.shape[0], len(self.operators.unaryvaribale_tensor)))
                unary_constrain_mask[unary_constrain_boolean] = (sequences[:, sequences.shape[1] - 1][:, None] == self.operators.unaryvaribale_tensor)[unary_constrain_boolean].float()
                unary_constrain_ones = torch.ones((output.shape))
                unary_constrain_ones[:, self.operators.unaryvaribale_tensor.long()] = unary_constrain_mask
                output = torch.minimum(output, unary_constrain_ones)

            # ~ Ensure the Descendants of trigonometric operators should not be trigonometric operators ~
            if (lengths[0].item() != 0.0):
                boolean_domain_begin = (sequences[:, sequences.shape[1] - 1][:, None] == self.operators.trigvaribale_tensor).any(dim=1)
                trig_indicate[boolean_domain_begin] = True
                boolean_domain_end = (parent_sibling[:, parent_sibling.shape[1] - 1][:, None] == self.operators.trigvaribale_tensor).any(dim=1)
                trig_indicate[boolean_domain_end] = False
                trig_constrain_mask = torch.ones((output.shape[0], len(self.operators.unaryvaribale_tensor)))
                trig_constrain_mask[trig_indicate] = 0
                trig_constrain_ones = torch.ones((output.shape))
                trig_constrain_ones[:, self.operators.trigvaribale_tensor.long()] = trig_constrain_mask
                output = torch.minimum(output, trig_constrain_ones)

            # ~ Ensure the children of an operator should not all be constants ~
            if (lengths[0].item() != 0.0):
                one_arity_constrain_boolean = (sequences[:, sequences.shape[1] - 1][:, None] == self.operators.arity_one).any(dim=1)
                sibling_constrain_boolean = (parent_sibling[:, parent_sibling.shape[1] - 1][:, None] == self.operators.const_tensor).any(dim=1)
                one_arity_constrain_mask = torch.ones((output.shape[0], len(self.operators.const_operators)))
                sibling_constrain_mask = torch.ones((output.shape[0], len(self.operators.const_operators)))
                one_arity_constrain_mask[one_arity_constrain_boolean] = 0
                sibling_constrain_mask[sibling_constrain_boolean] = 0
                const_constrain_mask = torch.minimum(sibling_constrain_mask, one_arity_constrain_mask)
                const_constrain_ones = torch.ones((output.shape))
                const_constrain_ones[:, self.operators.const_tensor.long()] = const_constrain_mask
                output = torch.minimum(output, const_constrain_ones)


        return output

    def get_parent_sibling(self, sequences, lengths):
        """ Subroutine 1 Computing parent and sibling inputs to the RNN
            Returns parent, sibling for the most recent token in token_list
        """
        parent_sibling = torch.ones((lengths.shape[0], 2)) * -1
        recent = int(lengths[0].item())-1

        c = torch.zeros(lengths.shape[0])
        for i in range(recent, -1, -1):
            # Determine arity of the i-th tokens
            token_i = sequences[:, i]
            arity = torch.zeros(lengths.shape[0])
            arity += torch.isin(token_i, self.operators.arity_two).long() * 2
            arity += torch.isin(token_i, self.operators.arity_one).long() * 1

            # Increment c by arity of the i-th token, minus 1
            c += arity
            c -= 1

            # In locations where c is zero (and parents and siblings that are -1),
            # we want to set parent_sibling to sequences[:, i] and sequeneces[:, i+1].
            # c_mask an n x 1 tensor that is True when c is zero and parents/siblings are -1.
            # It is False otherwise.
            c_mask = torch.logical_and(c==0, (parent_sibling == -1).all(axis=1))[:, None]

            # n x 2 tensor where dimension is 2 is sequences[:, i:i+1]
            # Since i+1 won't exist on the first iteration, we pad
            # (-1 is i+1 doesn't exist)
            i_ip1 = sequences[:, i:i+2]
            i_ip1 = F.pad(i_ip1, (0, 1), value=-1)[:, 0:2]

            # Set i_ip1 to 0 for indices where c_mask is False
            i_ip1 = i_ip1 * c_mask.long()

            # Set parent_sibling to 0 for indices where c_mask is True
            parent_sibling = parent_sibling * (~c_mask).long()

            parent_sibling = parent_sibling + i_ip1

        ###
        # If our most recent token has non-zero arity, then it is the
        # parent, and there is no sibling. We use the following procedure:
        ###

        # We create recent_nonzero_mask, an n x 1 tensor that is True if the most
        # recent token has non-zero arity and False otherwise.
        recent_nonzero_mask = (~torch.isin(sequences[:, recent], self.operators.arity_zero))[:, None]

        # This sets parent_sibling to 0 for instances where recent_nonzero_mask is True
        # If recent_nonzero_mask is False, the value of parent_sibling is unchanged
        parent_sibling = parent_sibling * (~recent_nonzero_mask).long()

        # This tensor is n x 2 where the 2 dimension is [recent token, -1]
        recent_parent_sibling = torch.cat((sequences[:, recent, None], -1*torch.ones((lengths.shape[0], 1))), axis=1)

        # We set values of recent_parent_sibling where recent_nonzero_mask is False
        # to zero.
        recent_parent_sibling = recent_parent_sibling * recent_nonzero_mask.long()

        # Finally, add recent_parent_sibling to parent_sibling.
        parent_sibling = parent_sibling + recent_parent_sibling


        return parent_sibling

    def get_next_input(self, parent_sibling):
        # Just convert -1 to 1 for now; it'll be zeroed out later
        parent = torch.abs(parent_sibling[:, 0]).long()
        sibling = torch.abs(parent_sibling[:, 1]).long()

        # Generate one-hot encoded tensors
        parent_onehot = F.one_hot(parent, num_classes=len(self.operators))
        sibling_onehot = F.one_hot(sibling, num_classes=len(self.operators))

        # Use a mask to zero out values that are -1. Parent should never be -1,
        # but we do it anyway.
        parent_mask = (~(parent_sibling[:, 0] == -1)).long()[:, None]
        parent_onehot = parent_onehot * parent_mask
        sibling_mask = (~(parent_sibling[:, 1] == -1)).long()[:, None]
        sibling_onehot = sibling_onehot * sibling_mask

        input_tensor = torch.cat((parent_onehot, sibling_onehot), axis=1)
        return input_tensor
