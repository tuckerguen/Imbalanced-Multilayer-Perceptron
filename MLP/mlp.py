import numpy as np
from MLP.layer import Layer


def i2h_deriv(nj, xji, dl, wo, xo):
    """
    dL/dwji (derivative of the loss w.r.t an
    edge) from an input node to a hidden node
    :param nj: Sum of input to hidden node
    :param dl: Derivative along edge from hidden to output
    :param wo: Weight along edge from hidden to output
    :param xo: Value along edge from hidden to output
    :return:
    """
    return (1 - np.tanh(nj) ** 2) * xji * dl * (wo / xo)


def h2o_deriv(nj, xji):
    """
    dL/dwji (derivative of the loss w.r.t an
    edge) from a hidden node to the output node
    :param nj: The sum of the inputs to the output node
    :param xji: The input along this edge
    :return: Derivative value
    """
    return (1 - np.tanh(nj) ** 2) * xji


class MLP:
    def __init__(self, n, h, _lambda, afcn):
        """
        Creates a multilayer perceptron with a single hidden layer
        and one output node, using a weighted sum loss function
        :param n: Number of input units
        :param h: Number of hidden units
        :param _lambda: Lambda value for weighted sum
        :param afcn: activation function of units
        """
        self.n = n
        self.h = h
        self.hidden = Layer(n, h, afcn)
        self.output = Layer(h, 1, afcn)
        self._lambda = _lambda

    def predict(self, x):
        """
        Prediction output of multilayer perceptron
        :param x: example to predict class of
        :return: value in (-1, 1)
        """
        return self.output.feed_forward(self.hidden.feed_forward(x))

    def err(self, T, y):
        return y - np.array([self.predict(x) for x in T])

    def weightsum(self, x1, x2):
        return self._lambda * x1 + (1 - self._lambda) * x2

    def loss(self, e1, e2):
        return self.weightsum(np.dot(np.transpose(e1), e1), np.dot(np.transpose(e2), e2))

    def Hess(self, jac1, jac2):
        return self.weightsum(np.dot(np.transpose(jac1), jac1), np.dot(np.transpose(jac2), jac2))

    def grad(self, Z1, e1, Z2, e2):
        return self.weightsum(np.dot(np.transpose(Z1), e1), np.dot(np.transpose(Z2), e2))

    def train(self, T1, T2):
        """
        Train the MLP on training examples
        :param T1: List of positive examples
        :param T2: List of negative examples
        :return: None
        """
        mu = 0.1
        beta = 10
        Jprev = float('inf')
        e1 = self.err(T1, 1)
        e2 = self.err(T2, -1)

        Jnew = self.loss(e1, e2)
        eps = 1e-6
        max_iters = 25000
        iters = 0

        # Main optimization loop
        while abs(Jprev - Jnew) > eps and iters < max_iters:
            # Perform levenberg marquardt optimization of the weights
            # Skip mu update for first iteration
            if not iters == 0:
                # Compute error vectors
                e1 = self.err(T1, 1)
                e2 = self.err(T2, -1)
                # Compute the loss for current weights
                Jprev = Jnew
                Jnew = self.loss(e1, e2)
                print(Jnew)
                # If loss decreased
                if Jnew < Jprev:
                    # update mu
                    mu /= beta
                # If loss increased
                else:
                    mu *= beta

            """update hidden-output weights"""
            # Compute jacobian for hidden-output weights
            # Convert weights from 2d array to 1d
            prev_inputs = self.output.ex_inputs
            prev_sums = self.output.ex_sums
            Z1o = np.array([[h2o_deriv(prev_sums[i][0], prev_inputs[i][j]) for j in range(self.h)]
                            for i in range(len(T1))])
            Z2o = np.array([[h2o_deriv(prev_sums[i][0], prev_inputs[i][j]) for j in range(self.h)]
                            for i in range(len(T1), len(T1)+len(T2))])
            # Compute hessian approximation
            H = self.Hess(Z1o, Z2o)
            # Compute gradient vector approximation
            g = self.grad(Z1o, e1, Z2o, e2)
            # Compute and apply weight update
            deltaw = np.dot(np.linalg.inv(np.add(H, (mu * np.identity(self.h)))), g)
            self.output.weights = self.output.weights - deltaw
                
            """update input-hidden weights"""
            # Compute jacobian for hidden-output weights
            h2o_weights = self.output.weights.ravel()
            prev_inputs_h = self.hidden.ex_inputs
            prev_sums_h = self.hidden.ex_sums

            Z1h = np.array([[i2h_deriv(prev_sums_h[x][h], prev_inputs_h[x][i], Z1o[x][h], h2o_weights[h], prev_inputs[x][h])
                       for i in range(self.n) for h in range(self.h)] for x in range(len(T1))])
            Z2h = np.array([[i2h_deriv(prev_sums_h[i][h], prev_inputs_h[x][i], Z2o[x-len(T1)][h], h2o_weights[h], prev_inputs[x][h])
                       for i in range(self.n) for h in range(self.h)] for x in range(len(T1), len(T1)+len(T2))])
            # Compute hessian approximation
            H = self.Hess(Z1h, Z2h)
            # Compute gradient vector approximation
            g = self.grad(Z1h, e1, Z2h, e2)
            # Apply weight update
            deltaw = np.dot(np.linalg.inv(np.add(H, (mu * np.identity(self.n * self.h)))), g)
            deltaw = deltaw.reshape(2, 3)
            self.hidden.weights = self.hidden.weights - deltaw

            # Reset stored sums and inputs
            self.output.ex_inputs = self.output.ex_sums = None
            self.hidden.ex_inputs = self.hidden.ex_sums = None

            iters += 1


"""
Section for the modified levenberg-marquadt learning rule
"""
# def Z(T1, w):
#
#
# def H(w, T1, T2):
#     Zt1 = Z(T1, w)
#     Zt2 = Z(T2, w)
#     return _lambda * np.dot(np.transpose(Zt1), Zt1) + (1 - _lambda) * np.dot(np.transpose(Zt2), Zt2)
#
# def wnew(wold, mu):
#     return wold - np.linalg.inv((H(wold) + mu * np.identity())) * g(wold)
