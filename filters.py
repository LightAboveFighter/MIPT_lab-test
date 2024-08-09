import numpy as np
from scipy.signal import convolve2d
import cv2

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for vert in range(Hi):
        for horiz in range(Wi):
            for i in range(Hk):
                for k in range(Wk):
                    if (not 0 <= vert - 1 + i < Hi) or (not 0 < horiz - 1 + k < Wi):
                        continue
                    out[vert][horiz] += image[vert - Hk//2 + i][horiz - Wk//2 + k] * kernel[-i - 1][-k - 1]

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2*pad_height, W + 2*pad_width))

    ### YOUR CODE HERE
    out[pad_height : H + pad_height, pad_width : W + pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    padded = zero_pad(image, Hk//2, Wk//2)
    reversed_kernel = kernel[::-1, ::-1]
    for vert in range(Hi):
        for horiz in range(Wi):
            out[vert][horiz] = np.sum( reversed_kernel * padded[vert : vert + Hk, horiz : horiz + Wk] )
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    out = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    # out = np.zeros_like(f)
    # Hf, Wf = f.shape
    ### YOUR CODE HERE
    # fft_x = np.fft.fft2(f)

    # y = zero_pad(g, Hf//2, Wf//2)
    # e = np.ones_like(g)
    # e = zero_pad(e, Hf//2, Wf//2)
    # x_y = np.fft.ifft2( conv_fast(fft_x, np.conjugate(np.fft.fft2(y))))
    # x_x = np.fft.ifft2( conv_fast( conv_fast(fft_x, np.conjugate( fft_x )), np.conjugate(np.fft.fft2(e) )))

    # out = (f + x_y) / f + x_x

    out = cv2.matchTemplate(f, g, method=0)
    

    # x_y = np.fft.ifft2(conv_fast(np.fft.fft2(f).real, np.fft.fft2(zero_pad(g[::-1, ::-1], Hf//2, Wf//2)).real )).real

    # p_x_y = np.fft.ifft2(conv_fast(np.fft.fft2(f).real, np.fft.fft2(zero_pad(g[::-1, ::-1], Hf//2, Wf//2)).real )).real
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
