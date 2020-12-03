import numpy as np
from mlp.layer import Layer
import matplotlib.pyplot as plt
from data_help.exset_ops import weighted_acc, accuracy
from _heapq import heappush, heappop


class MLP:
    def __init__(self, n, h, lambda_, afcn, mu, beta, repeat=False):
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
        self.mu = mu
        self.beta = beta
        self.hidden = Layer(n, h, afcn)
        self.output = Layer(h, 1, afcn)
        self.lambda_ = lambda_
        self.repeat = repeat

    def randomize_weights(self):
        self.hidden.randomize_weights()
        self.output.randomize_weights()

    def classify(self, x):
        return 1 if self.predict(x) > 0 else 0

    def confidence(self, x):
        return (self.predict(x) + 1) / 2.0

    def predict(self, x):
        """
        Prediction output of multilayer perceptron
        :param x: example to predict class of
        :return: value in (-1, 1)
        """
        return self.output.feed_forward(self.hidden.feed_forward(x))

    def err(self, T, y):
        return np.subtract(y, np.array([self.predict(x) for x in T]))

    def weightsum(self, x1, x2):
        return self.lambda_ * x1 + (1 - self.lambda_) * x2

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
        :return: list of x and loss values for plotting
        """
        mu = self.mu
        beta = self.beta

        mins0, x0s, l0s = self.LM_optimize_weights(mu, beta, T1, T2)

        if self.repeat:
            return self.repeat_optimization(mins0, x0s, l0s, T1, T2)
        else:
            return x0s, l0s

    def repeat_optimization(self, mins0, x0s, l0s, T1, T2):
        """
        Repeat LM optimization, with given initial weights
        :param mins0: list of tuples with loss, hidden weights and output weights
                      from original weight optimization
        :param x0s: list of x values for plotting loss
        :param l0s: list of loss values for plotting
        :param T1: majority class example set
        :param T2: minority class example set
        :return: final optimized list of x values and loss values for plotting
        """

        losses = []
        heappush(losses, (mins0, x0s, l0s))

        loss_values = np.array(l0s).reshape(len(x0s), )
        plt.plot(x0s, loss_values)
        plt.title(f"mu={self.mu}, beta={self.beta}")
        plt.show()

        # Repeat optimization for minima
        for i, weights in enumerate(mins0):
            print(f"Re-optimizing from loss at {-weights[0][0][0]:.3f}")
            self.hidden.weights = weights[1]
            self.output.weights = weights[2]
            mins, x1s, l1s = self.LM_optimize_weights(0.001, 3, T1, T2)

            loss_values1 = np.array(l1s).reshape(len(x1s), )
            plt.plot(x1s, loss_values1)
            plt.title(f"it={i}, mu={0.001}, beta={3}")
            plt.show()

            heappush(losses, (mins, x1s, l1s))

        best = losses[len(losses) - 1]
        self.hidden.weights = best[0][0][1]
        self.output.weights = best[0][0][2]
        return best[1], best[2]

    def LM_optimize_weights(self, mu, beta, T1, T2):
        """
        Run modified Levenberg-Marquardt optimization on the MLP weights
        :param mu: initial mu value
        :param beta: initial beta value
        :param T1: majority class example set
        :param T2: minority class example set
        :return: list of tuples of minimum visited losses and their weights,
                 x and loss arrays for plotting
        """

        # Define stopping conditions
        eps = 1e-4
        max_iters = 500
        iters = 0

        # Initialize loss values
        Jprev = float('inf')
        e1 = self.err(T1, 1)
        e2 = self.err(T2, -1)
        Jnew = self.loss(e1, e2)

        # Data tracking lists
        xs = []
        loss_values = []
        min_loss_and_weights = []

        # Main optimization loop
        while abs(Jprev - Jnew) > eps and iters < max_iters:
            # Perform levenberg marquardt optimization of the weights

            # Store current iteration loss and number
            loss_values.append(Jnew)
            xs.append(iters)

            # Skip mu update for first iteration
            if not iters == 0:
                # Compute error vectors
                e1 = self.err(T1, 1)
                e2 = self.err(T2, -1)
                # Compute the loss for current weights
                Jprev = Jnew
                Jnew = self.loss(e1, e2)
                # Report current loss
                print(Jnew, end='\r')

                # Track top 3 minimum losses visited
                try:
                    heappush(min_loss_and_weights, (-Jnew, self.hidden.weights, self.output.weights))
                    if len(min_loss_and_weights) > 3:
                        heappop(min_loss_and_weights)
                except ValueError:
                    print("Loss storage failed")

                # If loss decreased
                if Jnew < Jprev and Jnew != Jprev:
                    # update mu
                    mu /= beta
                # If loss increased
                else:
                    mu *= beta

            # update hidden-output weights
            Z1o, Z2o = self.update_output_weights(e1, e2, mu, len(T1), len(T2))

            # update input-hidden weights
            self.update_hidden_weights(e1, e2, mu, Z1o, Z2o, len(T1), len(T2))

            # Reset stored sums and inputs
            self.output.ex_inputs = self.output.ex_sums = None
            self.hidden.ex_inputs = self.hidden.ex_sums = None

            iters += 1

        if iters == max_iters:
            print("Did not converge")

        print("Final loss=", Jnew)
        return min_loss_and_weights, xs, loss_values

    def update_hidden_weights(self, e1, e2, mu, Z1o, Z2o, N1, N2):
        """
        Perform a single update step of the input-hidden weights of the MLP
        :param e1: error list on T1
        :param e2: error list on T2
        :param mu: current mu
        :param Z1o: T1 Jacobian of output weights
        :param Z2o: T2 Jacobian of output weights
        :param N1: Length of T1
        :param N2: Length of T2
        :return:
        """
        # Compute jacobian for hidden-output weights
        prev_inputs = self.output.ex_inputs
        h2o_weights = self.output.weights.ravel()
        prev_inputs_h = self.hidden.ex_inputs
        prev_sums_h = self.hidden.ex_sums

        Z1h = np.array(
            [[i2h_deriv(prev_sums_h[x][h], prev_inputs_h[x][i], Z1o[x][h], h2o_weights[h], prev_inputs[x][h])
              for i in range(self.n) for h in range(self.h)] for x in range(N1)])
        Z2h = np.array([[i2h_deriv(prev_sums_h[i][h], prev_inputs_h[x][i], Z2o[x - N1][h], h2o_weights[h],
                                   prev_inputs[x][h])
                         for i in range(self.n) for h in range(self.h)] for x in range(N1, N1 + N2)])

        # Compute hessian approximation
        H = self.Hess(Z1h, Z2h)
        # Compute gradient vector approximation
        g = self.grad(Z1h, e1, Z2h, e2)

        # Apply weight update
        deltaw = np.dot(np.linalg.inv(np.add(H, (mu * np.identity(self.n * self.h)))), g)
        deltaw = deltaw.reshape(self.n, self.h)
        self.hidden.weights = np.add(self.hidden.weights, deltaw)

    def update_output_weights(self, e1, e2, mu, N1, N2):
        """
        Perform a single update step of the hidden to output weights of the MLP
        :param e1: error list on T1
        :param e2: error list on T2
        :param N1: Length of T1
        :param N2: Length of T2
        :param mu: current mu
        :return:
        """
        # Convert weights from 2d array to 1d
        prev_inputs = self.output.ex_inputs
        prev_sums = self.output.ex_sums

        # Compute jacobian for hidden-output weights
        Z1o = np.array([[h2o_deriv(prev_sums[i][0], prev_inputs[i][j]) for j in range(self.h)]
                        for i in range(N1)])
        Z2o = np.array([[h2o_deriv(prev_sums[i][0], prev_inputs[i][j]) for j in range(self.h)]
                        for i in range(N1, N1 + N2)])

        # Compute hessian approximation
        H = self.Hess(Z1o, Z2o)
        # Compute gradient vector approximation
        g = self.grad(Z1o, e1, Z2o, e2)

        # Compute and apply weight update
        deltaw = np.dot(np.linalg.inv(np.add(H, (mu * np.identity(self.h)))), g)
        deltaw = deltaw.flatten('C')
        self.output.weights = np.add(self.output.weights, deltaw)

        return Z1o, Z2o

    def plot_decision_boundary(self):
        print("Plotting decision boundary")
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((200, 200))

        for idx, i in enumerate(x):
            for idy, j in enumerate(y):
                Z[idx][idy] = self.predict([i, j]) > 0

        plt.pcolormesh(X, Y, Z)


def i2h_deriv(nj, xji, dl, wo, xo):
    """
    dL/dwji (derivative of the loss w.r.t an
    edge) from an input node to a hidden node
    :param xji: input along edge from input to hidden node
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


def eval_mlp(mlp, T, T1, T2):
    pred1 = [-1 if mlp.predict(ex) < 0 else 1 for ex in T1]
    pred2 = [-1 if mlp.predict(ex) < 0 else 1 for ex in T2]
    pred = np.append(pred1, pred2)
    act1 = [1] * len(T1)
    act2 = [-1] * len(T2)
    act = np.append(act1, act2)

    maj_acc = accuracy(pred1, act1)
    min_acc = accuracy(pred2, act2)
    lambda_weight_acc = mlp.weightsum(accuracy(pred1, act1), accuracy(pred2, act2))
    overall_acc = accuracy(pred, act)
    w_acc = weighted_acc(pred, act)
    gmean = np.sqrt(maj_acc * min_acc)

    return [maj_acc, min_acc, lambda_weight_acc, overall_acc, w_acc, gmean]


def print_accs(acc):
    print("Majority class accuracy:", acc[0])
    print("Minority class accuracy:", acc[1])
    print("Overall weighted accuracy", acc[2])
    print("Overall accuracy", acc[3])
    print("Weighted accuracy", acc[4])
    print("Gmean accuracy", acc[5])
