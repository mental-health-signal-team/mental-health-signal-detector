import pandas as pd

from src.training.train import create_X_y, split_data, vectorize_data


def test_create_X_y():
	df = pd.DataFrame(
		{
			"clean_title": ["i feel good", "i feel bad", "need help"],
			"label": [0, 1, 1],
			"extra": [10, 20, 30],
		}
	)

	X, y = create_X_y(df)

	assert list(X) == ["i feel good", "i feel bad", "need help"]
	assert list(y) == [0, 1, 1]


def test_split_data():
	X = pd.Series([f"text_{i}" for i in range(10)])
	y = pd.Series([0, 1] * 5)

	X_train_1, X_test_1, y_train_1, y_test_1 = split_data(
		X, y, test_size=0.3, random_state=42
	)

	assert len(X_train_1) == 7
	assert len(X_test_1) == 3
	assert len(y_train_1) == 7
	assert len(y_test_1) == 3


def test_vectorize_data():
	X_train = pd.Series(["happy day", "very happy", "sad day"])
	X_test = pd.Series(["happy", "unknownword"])

	X_train_vec, X_test_vec, vectorizer = vectorize_data(
		X_train, X_test, max_features=5
	)

	assert X_train_vec.shape[0] == 3
	assert X_test_vec.shape[0] == 2
	assert X_train_vec.shape[1] <= 5
	assert X_test_vec.shape[1] == X_train_vec.shape[1]
	assert "happy" in vectorizer.vocabulary_
	assert "unknownword" not in vectorizer.vocabulary_
	assert X_test_vec[1].nnz == 0
