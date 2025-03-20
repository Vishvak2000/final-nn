from nn.nn import NeuralNetwork
import numpy as np
from nn.preprocess import sample_seqs, one_hot_encode_seqs

def test_single_forward():
    # Create a simple 2-layer neural network
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                 {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='binary_cross_entropy'
    )
    
    X = np.array([[1, 2, 3]]).T  # Shape: [3, 1]
    W_curr = nn._param_dict['W1']
    b_curr = nn._param_dict['b1']
    
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, X, 'relu')
    
    # Test the shape of the output
    assert A_curr.shape == (2, 1), f"Expected A_curr shape (2, 1), got {A_curr.shape}"
    assert Z_curr.shape == (2, 1), f"Expected Z_curr shape (2, 1), got {Z_curr.shape}"


def test_forward():
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                 {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='binary_cross_entropy'
    )
    
    X = np.array([[1, 2, 3]])  # Shape: [3, 1]
    output, cache = nn.forward(X)
    
    # Test the output shape
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
    
    # Check that cache contains all required values
    assert 'A0' in cache, "Cache missing A0"
    assert 'Z1' in cache, "Cache missing Z1"
    assert 'A1' in cache, "Cache missing A1"
    assert 'Z2' in cache, "Cache missing Z2"
    assert 'A2' in cache, "Cache missing A2"

def test_single_backprop():
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                 {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='binary_cross_entropy'
    )
    
    X = np.array([[1, 2, 3]])  # Shape: [3, 1]
    y = np.array([[1]])      # Shape: [1, 1]
    y_hat = nn.forward(X)[0]     # Get predictions
    cache = nn.forward(X)[1]     # Cache for backprop
    
    # Get gradients using backprop
    grad_dict = nn.backprop(y, y_hat, cache)
    
    # Check gradients exist for each layer
    assert 'dW1' in grad_dict, "Missing dW1 in gradient dictionary"
    assert 'db1' in grad_dict, "Missing db1 in gradient dictionary"
    assert 'dW2' in grad_dict, "Missing dW2 in gradient dictionary"
    assert 'db2' in grad_dict, "Missing db2 in gradient dictionary"


def test_predict():
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                    {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='binary_cross_entropy'
    )

    X = np.array([[1, 2, 3]])  # Shape: [3, 1]
    y_hat = nn.predict(X)

    # Test the prediction shape
    assert y_hat.shape == (1, 1), f"Expected y_hat shape (1, 1), got {y_hat.shape}"

def test_binary_cross_entropy():
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                 {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='binary_cross_entropy'
    )
    
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    
    loss = nn._binary_cross_entropy(y, y_hat)
    
    # Check that the loss is a scalar value
    assert isinstance(loss, float), f"Expected loss type float, got {type(loss)}"
    assert loss > 0, f"Loss should be positive, got {loss}"

def test_binary_cross_entropy_backprop():
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                 {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='binary_cross_entropy'
    )
    
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    
    # Check that the gradient has the correct shape
    assert dA.shape == y_hat.shape, f"Expected gradient shape {y_hat.shape}, got {dA.shape}"


def test_mean_squared_error():
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                 {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='mean_squared_error'
    )
    
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    
    loss = nn._mean_squared_error(y, y_hat)
    
    # Check that the loss is a scalar value
    assert isinstance(loss, float), f"Expected loss type float, got {type(loss)}"
    assert loss > 0, f"Loss should be positive, got {loss}"

def test_mean_squared_error_backprop():
    nn = NeuralNetwork(
        nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                 {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01,
        seed=42,
        batch_size=1,
        epochs=10,
        loss_function='mean_squared_error'
    )
    
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    
    dA = nn._mean_squared_error_backprop(y, y_hat)
    
    # Check that the gradient has the correct shape
    assert dA.shape == y_hat.shape, f"Expected gradient shape {y_hat.shape}, got {dA.shape}"


def test_sample_seqs():
    
    seqs = ["ATCG", "GCTA", "TTAG", "CCGA", "GGTT"]
    labels = [1, 0, 1, 0, 1]  # Class imbalance (3 positives, 2 negatives)

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # Test that class distribution is balanced
    assert sampled_labels.count(1) == sampled_labels.count(0), "Classes are not balanced"

    # Test that sequences are all the same length
    seq_lengths = set(len(seq) for seq in sampled_seqs)
    assert len(seq_lengths) == 1, f"Sequences have varying lengths: {seq_lengths}"

    # Test that sampled sequences are from the original set (or padded versions)
    for seq in sampled_seqs:
        original_or_padded = any(seq.startswith(orig_seq) for orig_seq in seqs)
        assert original_or_padded, f"Unexpected sequence found: {seq}"

def test_one_hot_encode_seqs():
    seqs = ["AA", "TC", "CG", "GC"]


    encoded = one_hot_encode_seqs(seqs)

    # Expected one-hot encodings
    expected = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 0, 1, 0]])

    assert np.array_equal(encoded, expected), f"One-hot encoding incorrect: {encoded}"

    # Test that all sequences produce a one-hot encoding 4x their length
    for i, seq in enumerate(seqs):
        assert len(encoded[i]) == 4 * len(seq), f"Incorrect encoding length for {seq}"