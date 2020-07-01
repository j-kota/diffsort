import torch
import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.linalg import block_diag


class softRank(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        # the input is a batch of tensors (theta) to rank
        thetas = input.detach().numpy()

        # get a list of forward (rankings) and backward (jacobian) values per sample
        results_list = [ fwdRank(theta) for theta in thetas ]

        # separate the fwd and bkwd results
        ranks_list   = [ torch.Tensor( a ) for (a,b) in results_list ]
        jacobi_list  = [ torch.Tensor( b ) for (a,b) in results_list ]

        # the results are stacked to conform to batch processing format (first dim holds batch)
        ctx.save_for_backward(  torch.stack(jacobi_list)  )
        return torch.stack( ranks_list )

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        jacobians, = ctx.saved_tensors
        N = len(grad_output)

        #print("inside softRank.backward():")
        #print("2-norm of jacobians = ")
        #print( jacobians.norm(2,0) )
        
        # multiply each gradient by the jacobian for the corresponding sample
        # then restack the results to preserve the batch gradients' format
        grad_input = torch.stack( [ torch.matmul( grad_output[i] , jacobians[i] ) for i in range(0,N) ] )
        
        return grad_input


# numpy function that carries out
# the forward algorithm and calculates jacobians
# for the backward pass
# returns both
def fwdRank(theta):

    # Define variables, constants and permutations
    # with notation as in the original paper
    rho = np.arange(len(theta))[::-1] + 1
    eps = (1.0 / 10.0)
    x = np.arange(len(theta))[::-1]
    z = -theta/eps
    w = rho
    sigma = np.argsort(theta)
    sigma_inv = np.argsort(sigma)
    s = z[sigma]

    ir = IsotonicRegression()

    # v is the solution to isotonic regression over s-w
    v = ir.fit_transform(x,  s-w)

    # This is the final result of the ranking operator
    answer = z - v[sigma_inv]

    # blocks will hold sequential integers, constant over
    # each block of the isotonic regression
    c = 0
    blocks = [0]*len(v)
    for i in range(1,len(v)):
        if v[i] == v[i-1]:
            blocks[i] = blocks[i-1]
        else:
            c += 1
            blocks[i] = c

    # compute |B_j|, the length of each block
    blocklens = [1]
    for i in range(1,len(blocks)):
        if blocks[i] == blocks[i-1]:
            blocklens[-1] +=1
        else:
            blocklens.append(1)

    # compute the nonzero jacobian values
    # they are the lengths of the blocks/intervals on which v is constant
    num_blocks = len(blocklens)
    blockvals = 1.0/np.array(blocklens)

    # create the block matrices that will occupy
    # the (block) diagonal of the jacobian of v
    jacobi_v_blox = [ np.ones( (blocklens[i],blocklens[i]) )*blockvals[i] for i in range(0,num_blocks) ]
    jacobi_len = sum( blocklens )

    # assemble v's jacobian operator from the blocks the from its diagonal
    jacobi_v = block_diag( *jacobi_v_blox )

    # jacobian operator for P wrt variable z = -theta/eps
    # The I is absorbed into the permutations due to symmetry
    # The inner permutation is due to the chain rule between z and s
    # The outer permutation is due to Proposition 3
    I = np.identity( jacobi_v.shape[0] )
    jacobi_P = ( (I-jacobi_v)[sigma_inv] )[:,sigma_inv]  
    
    # jacobian operator for P wrt the original input theta ( using chain rule: dz/dtheta = -1/eps*I )
    jacobi_P_theta = jacobi_P * (-1.0/eps)

    return  answer ,  jacobi_P_theta 




    
if __name__ == "__main__":

    #testing 

    torch.set_printoptions(precision=10)
    sr = softRank()
    

    theta = torch.Tensor( [1.0,  5.0, 1000.0,  55.0, 3.0] )
    theta.requires_grad = True
    print("theta.requires_grad: ", theta.requires_grad)
    
    ranks =  sr.apply(theta)
    
    ranksum = torch.sum(ranks)
    b = ranksum.backward()

    print("ranks = ", ranks)
    print("ranks.grad = ", ranks.grad)
    print("ranksum.grad = ", ranksum.grad)
    print("return value of backward():", b)
    
    

    
