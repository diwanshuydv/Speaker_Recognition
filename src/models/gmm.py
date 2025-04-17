from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

# Load data
x = np.load("./../../data/features/x_1_3.npy")
y = np.load("./../../data/features/y_1_3.npy")

# Encode labels if they are strings
le = LabelEncoder()
y = le.fit_transform(y)

# Flatten the features if needed
x = x.reshape(x.shape[0], -1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print("Training Data Shape:", X_train.shape)

# Number of classes
n_classes = np.unique(y_train).shape[0]


def fit_gmm_for_class(class_idx):
    """Fit a GMM to samples of a specific class."""
    gmm = GaussianMixture(
        n_components=5, covariance_type='full', random_state=42
    )
    gmm.fit(X_train[y_train == class_idx])
    print(f"Fitted GMM for class {class_idx}")
    return gmm


# Fit a separate GMM for each class using parallel processing
gmms = Parallel(n_jobs=-1)(
    delayed(fit_gmm_for_class)(i) for i in range(n_classes)
)


def score_samples(gmm):
    """Score test samples using the given GMM."""
    return gmm.score_samples(X_test)


# Compute log-likelihoods
log_likelihoods = np.array(
    Parallel(n_jobs=4)(delayed(score_samples)(gmm) for gmm in gmms)
).T

# Predict the class with the highest likelihood
y_pred = np.argmax(log_likelihoods, axis=1)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("GMM Classification Accuracy:", accuracy)
