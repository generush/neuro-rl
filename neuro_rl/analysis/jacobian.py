import torch
from torch.autograd.functional import jacobian

def compute_jacobian(rnn, input, hidden):

    HIDDEN_DIM = hidden.size(dim=1) // 2

    h0 = hidden[:,:HIDDEN_DIM]
    c0 = hidden[:,HIDDEN_DIM:]

    # Make sure the hidden state requires gradient
    h0.requires_grad_(True)
    c0.requires_grad_(True)
    input.requires_grad_(True)

    # Concatenate h0 and c0 along a new dimension to create a single tensor
    hc = torch.cat((h0.unsqueeze(0), c0.unsqueeze(0)), dim=0)

    # Define a function for the LSTM with respect to input
    def func_input(input):
        _, (h_out, c_out) = rnn(input, (h0, c0))
        return torch.cat((h_out, c_out), dim=0)

    # Define a function for the LSTM with respect to hidden states
    def func_hidden(hc):
        h0, c0 = torch.split(hc, 1, dim=0)
        h0 = h0.squeeze(0)
        c0 = c0.squeeze(0)
        _, (h_out, c_out) = rnn(input, (h0, c0))
        return torch.cat((h_out, c_out), dim=0)

    # Compute the Jacobian for the output wrt the input
    jacobian_matrix_input = jacobian(func_input, input)

    # j_hc = del_h_new / del_c
    J_hi = torch.squeeze(jacobian_matrix_input)[0,:,:].squeeze()
    J_ci = torch.squeeze(jacobian_matrix_input)[1,:,:].squeeze()
    J_input = torch.cat((J_hi, J_ci), dim=0)

    # Compute the Jacobian for the output wrt the hidden states
    jacobian_matrix_hidden = jacobian(func_hidden, hc)

    # j_hc = del_h_new / del_c
    J_hh = torch.squeeze(jacobian_matrix_hidden)[0,:,0,:].squeeze()
    J_hc = torch.squeeze(jacobian_matrix_hidden)[0,:,1,:].squeeze()
    J_ch = torch.squeeze(jacobian_matrix_hidden)[1,:,0,:].squeeze()
    J_cc = torch.squeeze(jacobian_matrix_hidden)[1,:,1,:].squeeze()
    J_hidden = torch.cat(
        (
            torch.cat((J_hh, J_hc), dim=1), 
            torch.cat((J_ch, J_cc), dim=1)
        ), 
        dim=0
    )

    return J_input, J_hidden


def compute_jacobian_alternate(rnn, input, hidden):

    HIDDEN_DIM = hidden.size(dim=1) // 2

    hx = hidden[:,:HIDDEN_DIM].requires_grad_(True)
    cx = hidden[:,HIDDEN_DIM:].requires_grad_(True)

    # Make sure the hidden state requires gradient
    input.requires_grad_(True)
    J_hh = torch.zeros(HIDDEN_DIM, HIDDEN_DIM)
    J_hc = torch.zeros(HIDDEN_DIM, HIDDEN_DIM)
    J_ch = torch.zeros(HIDDEN_DIM, HIDDEN_DIM)
    J_cc = torch.zeros(HIDDEN_DIM, HIDDEN_DIM)
    J_hi = torch.zeros(HIDDEN_DIM, INPUT_DIM)
    J_ci = torch.zeros(HIDDEN_DIM, INPUT_DIM)

    for i in range(HIDDEN_DIM):
        output = torch.zeros(1, HIDDEN_DIM)
        output[:, i] = 1

        _, (hx_new, cx_new) = a_rnn(input, (hx, cx))  # LSTM returns output, (h_n, c_n)

        g_hh = torch.autograd.grad(hx_new, hx, grad_outputs=output, retain_graph=True)[0]
        g_hc = torch.autograd.grad(hx_new, cx, grad_outputs=output, retain_graph=True)[0]
        g_ch = torch.autograd.grad(cx_new, hx, grad_outputs=output, retain_graph=True)[0]
        g_cc = torch.autograd.grad(cx_new, cx, grad_outputs=output, retain_graph=True)[0]
        J_hh[i,:] = g_hh
        J_hc[i,:] = g_hc
        J_ch[i,:] = g_ch
        J_cc[i,:] = g_cc

        g_hi = torch.autograd.grad(hx_new, input, grad_outputs=output, retain_graph=True)[0]
        g_ci = torch.autograd.grad(cx_new, input, grad_outputs=output, retain_graph=True)[0]
        J_hi[i,:] = g_hi
        J_ci[i,:] = g_ci

    J_hidden = torch.cat(
        (
            torch.cat((J_hh, J_hc), dim=1), 
            torch.cat((J_ch, J_cc), dim=1)
        ), 
        dim=0
    )

    J_input = torch.cat((J_hi, J_ci), dim=0)

    return J_input, J_hidden
