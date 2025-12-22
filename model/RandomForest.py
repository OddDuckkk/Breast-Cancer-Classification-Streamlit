import numpy as np
import pandas as pd
from collections import Counter

class Node:
    """Representasi satu simpul (node) dalam pohon keputusan."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Indeks fitur yang dipakai untuk split (misal: fitur ke-2)
        self.threshold = threshold  # Nilai ambang batas (misal: <= 15.5)
        self.left = left            # Cabang kiri (True)
        self.right = right          # Cabang kanan (False)
        self.value = value          # Nilai prediksi (Hanya ada di Leaf Node)

    def is_leaf_node(self):
        return self.value is not None
    
class ManualDecisionTree:
    """
    Decision Tree Classifier from scratch.
    - Menggunakan Gini Impurity untuk split.
    - Menangani fitur numerik kontinu.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    # --- PROSES TRAINING (FIT) ---
    def fit(self, X, y):
        # Jika n_features tidak diset, gunakan semua fitur (default)
        # Nanti ini akan di-override oleh Random Forest untuk seleksi fitur acak
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # 1. Cek Kriteria Berhenti (Stop Conditions)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 2. Random Feature Selection (Ciri khas RF ada di sini juga)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # 3. Cari Split Terbaik (Greedy Search)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Jika tidak ketemu split yang bagus, jadikan leaf
        if best_feat is None:
             leaf_value = self._most_common_label(y)
             return Node(value=leaf_value)

        # 4. Lakukan Split Data
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # Cek tiap nilai unik

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Parent Gini
        parent_gini = self._gini(y)

        # Generate Split
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Weighted Average Child Gini
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * e_l + (n_r / n) * e_r

        # Gain = Pengurangan Impurity
        ig = parent_gini - child_gini
        return ig

    def _split(self, X_column, split_thresh):
        # Pisahkan data: Kiri <= Threshold, Kanan > Threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        # Rumus Gini Impurity Manual: 1 - sum(p^2)
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _most_common_label(self, y):
        if len(y) == 0: return 0
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    # --- PROSES TESTING (PREDICT) ---
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
class ManualRandomForest:
    """
    Random Forest from scratch.
    - Menggunakan Bootstrap Sampling (Bagging).
    - Menggunakan Random Feature Selection.
    - Menggunakan Majority Voting.
    """
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _to_numpy(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.asarray(X)

    # --- TAHAP TRAINING ---
    def fit(self, X, y):
        X = self._to_numpy(X)
        y = self._to_numpy(y)
        self.trees = []

        for _ in range(self.n_estimators):
            # A. Bootstrap Sampling (Bagging)
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            # B. Hitung jumlah fitur acak (Random Feature Selection)
            n_features = X.shape[1]
            if self.max_features == 'sqrt':
                n_features_subset = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                n_features_subset = int(np.log2(n_features))
            else:
                n_features_subset = n_features

            # C. Bangun Pohon (Training Decision Tree)
            tree = ManualDecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=n_features_subset
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

# --- TAHAP TESTING ---
    def predict(self, X):
        X = self._to_numpy(X)

        # 1. Kumpulkan prediksi dari semua pohon
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # 2. Transpose agar struktur jadi: [n_sampel, n_pohon]
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # 3. Majority Voting
        y_pred = []
        for preds in tree_preds:
            # Cari modus (nilai yang paling sering muncul)
            counter = Counter(preds)
            y_pred.append(counter.most_common(1)[0][0])

        return np.array(y_pred)