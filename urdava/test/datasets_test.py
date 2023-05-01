from urdava.dataset import load_dataset

X, y = load_dataset('phoneme')
assert X.shape[0] == y.shape[0], X
