import random
import math

# Simple helper functions for vector operations (without numpy)

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def vec_add(a, b):
    return [x+y for x, y in zip(a, b)]

def mat_vec_mul(mat, vec):
    return [dot(row, vec) for row in mat]

class Cell:
    """A small neural network cell with tanh activation."""
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = [[(random.random() - 0.5) * 0.1 for _ in range(input_size)]
                        for _ in range(output_size)]
        self.bias = [0.0] * output_size

    def forward(self, x):
        z = vec_add(mat_vec_mul(self.weights, x), self.bias)
        return [math.tanh(v) for v in z]

    def update(self, x, grad_out, lr=0.01):
        # derivative of tanh
        grad_activation = [g * (1 - y*y) for g, y in zip(grad_out, self.forward(x))]
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weights[i][j] -= lr * grad_activation[i] * x[j]
            self.bias[i] -= lr * grad_activation[i]
        # return gradient for previous layer
        grad_in = []
        for j in range(self.input_size):
            grad_sum = sum(self.weights[i][j] * grad_activation[i]
                           for i in range(self.output_size))
            grad_in.append(grad_sum)
        return grad_in

class CellModel:
    """A chain of cells representing the whole model."""
    def __init__(self, sizes):
        self.cells = [Cell(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def forward(self, x):
        for cell in self.cells:
            x = cell.forward(x)
        return x

    def train(self, samples, epochs=10, lr=0.05):
        for epoch in range(epochs):
            total_loss = 0.0
            for inp, target in samples:
                # forward pass
                outputs = [inp]
                x = inp
                for cell in self.cells:
                    x = cell.forward(x)
                    outputs.append(x)
                # compute loss (MSE)
                loss_grad = [x_i - t_i for x_i, t_i in zip(x, target)]
                total_loss += sum(g*g for g in loss_grad) / len(loss_grad)
                # backpropagate
                grad = loss_grad
                for idx in range(len(self.cells)-1, -1, -1):
                    grad = self.cells[idx].update(outputs[idx], grad, lr=lr)
            print(f"Epoch {epoch+1}, loss {total_loss/len(samples):.4f}")

if __name__ == "__main__":
    # example usage
    # Map characters to simple vectors (one-hot)
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    token_to_id = {ch: i for i, ch in enumerate(vocab)}
    vocab_size = len(vocab)
    def one_hot(idx):
        return [1.0 if i == idx else 0.0 for i in range(vocab_size)]

    # simple training data: learn to map input char to the same output char
    samples = []
    for ch in vocab:
        idx = token_to_id[ch]
        samples.append((one_hot(idx), one_hot(idx)))

    model = CellModel([vocab_size, 16, vocab_size])
    model.train(samples, epochs=20, lr=0.1)
    # test
    test_input = one_hot(token_to_id['h'])
    out = model.forward(test_input)
    predicted = vocab[out.index(max(out))]
    print(f"Input h -> predicted {predicted}")
