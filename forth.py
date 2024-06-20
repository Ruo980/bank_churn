import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# 创建一个二分类问题的数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
log_reg = LogisticRegression().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)
gb = GradientBoostingClassifier().fit(X_train, y_train)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)

# 定义绘图函数
def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)

# 绘制决策边界
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Decision Boundaries for Different Models', fontsize=16)

plot_decision_boundary(log_reg, X_test, y_test, axes[0, 0], "Logistic Regression")
plot_decision_boundary(rf, X_test, y_test, axes[0, 1], "Random Forest")
plot_decision_boundary(gb, X_test, y_test, axes[1, 0], "Gradient Boosting")
plot_decision_boundary(xgb_model, X_test, y_test, axes[1, 1], "XGBoost")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
