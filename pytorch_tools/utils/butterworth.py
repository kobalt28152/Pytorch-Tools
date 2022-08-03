import numpy as np

# Butterworth filter:
#   Used to soften the predictions along the edges of the image
def D(u, v, P, Q):
    """ Distance function

    Distance from (u, v) to the image center (P/2, Q/2).

    Parameters
    ----------
    u : int
        u position (x position in the image)
    v : int
        v position (y position in the image)
    P : int
        image size (width)
    Q : int
        image size (height)

    Returns
    -------
    float
        distance
    """
    return np.sqrt((u-P/2)**2 + (v-Q/2)**2)

def Butterworth(D0, P, Q, n=1):
    """ Butterworth filter """
    u = np.arange(P)
    v = np.arange(Q)
    uu, vv = np.meshgrid(u, v)
    D_uv = D(uu, vv, P, Q)
    return 1 / (1 + (D_uv/D0)**(2*n))


