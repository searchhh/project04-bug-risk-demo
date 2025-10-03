
import pandas as pd
from sklearn.datasets import make_classification

print("使用 sklearn的make_classification 生成模拟数据...")

# ---- 核心参数 ----
N_SAMPLES = 10000        # 总样本数
N_FEARTURES = 3         # 特征总数 （对应churn, dev_count, sync_keywords
N_INFORMATIVE = 2       # 有效特征数 (假设3个特征里有2个真正和bug相关)
N_REDUNDANT = 1         # 冗余特征数 模拟现实里的多重共线性（高度相关的特征）。
# 模型（尤其是线性模型）会更难判别每个特征的独立贡献；树模型影响较小，但特征重要性会被“稀释”。
CLASS_WEIGHT = [0.9, 0.1]  # 90%是clean， 10%是Buggy

# ----生成数据----
# make_classification 会返回特征矩阵X 和标签向量y
# random_state 保证每次生成的数据都一样

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEARTURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_repeated=0,    # 生成重复特征：从前面已生成的特征里原样复制一份（完全一样），就是一模一样的列。
    # 作用：模拟重复字段/重复编码的脏数据；对大多数模型没有增益，还可能扰乱特征选择与重要性评估。
    n_classes=2,    # 二分类
    n_clusters_per_class=1,  # 每一个类是个紧凑的团
    weights=CLASS_WEIGHT,
    flip_y=0.01,     # 模拟噪音
    class_sep=0.8  # 簇之间的距离尺度， 越大越容易区分
)

# ---- 转换为Pandas DataFream ----
feature_names = ['churn', 'dev_count', 'sync_keywords']
df = pd.DataFrame(X, columns=feature_names)
df['is_buggy'] = y

# 原始生成的数据是标准化的 （均值0， 方差1）， 我们可以把他缩放到一个更直观的范围
df['churn'] = abs(df['churn'] * 150 + 200).astype(int)
df['dev_count'] = abs(df['dev_count'] * 5 + 7).astype(int)
df['sync_keywords'] = abs(df['sync_keywords'] * 8 + 4).astype(int)

# ----保存到 CSV ----
df.to_csv('.\data\data.csv', index=False)

print("\n 模拟数据 'data.csv' 已经生成")
print(f"数据形状 f{df.shape}")
print(f"数据中 Buggy 的样本比例: {df['is_buggy'].mean():.2%}")
print("\n 数据前五行预览")
print(df.head())
