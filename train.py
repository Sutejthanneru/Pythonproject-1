import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter error

# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score

# Load dataset (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize and save the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.savefig('decision_tree.png')  # Save the plot instead of showing it
print("Decision tree saved as 'decision_tree.png'")

# Save the trained model
joblib.dump(clf, 'iris_decision_tree.pkl')
print("Model saved successfully!")
