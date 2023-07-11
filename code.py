import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def print_line():
    print("=" * 60)

#获取数据集的地方
import os
path=r'E:/soft/PyCharm/work'
print(os.listdir(path+"/input"))
#任何写入当前目录的结果都保存为输出。
#['train.csv', 'test.csv']


#(1)DATA PREPROCESSING数据预处理
print_line()
print("(1)数据预处理")
df = pd.read_csv(path+'/input/train.csv') #读取数据train
test = pd.read_csv(path+'/input/test.csv')
print(df.head())   #需要print才能够输出

# 检查是否有缺失的值,对于列
df.isnull().sum().max()
print(df.columns)

#目标价值分析
#理解预测值，它是热编码的，在现实生活中价格不会被热编码（一位有效编码）。
mid=df['price_range'].describe(), df['price_range'].unique()
print(mid)
#output:mean  std  min~max,将价格0-3变为5档

#数组的数字用热力图的颜色值
corrmat = df.corr()
f,ax = plt.subplots(figsize=(12,10))
sns.heatmap(corrmat,vmax=0.8,square=True,annot=True,annot_kws={'size':8})
plt.plot(corrmat)
#plt.show()

#很明显，我们可以看到每个类别都有不同的值范围集
f, ax = plt.subplots(figsize=(10,4))
plt.scatter(y=df['price_range'],x=df['battery_power'],color='red')
plt.scatter(y=df['price_range'],x=df['ram'],color='Green')
plt.scatter(y=df['price_range'],x=df['n_cores'],color='blue')
plt.scatter(y=df['price_range'],x=df['mobile_wt'],color='orange')
plt.scatter(y=df['price_range'],x=df['pc'],color='pink')
plt.show()

#现在预处理结束，在数据集中不需要创建虚拟变量或处理丢失的数据，
# 因为数据集没有任何丢失的数据

#(2)支持向量机及方法
print_line()
print("(2)支持向量机及方法")

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#x是train数据集
y_t = np.array(df['price_range'])
X_t = df
X_t = df.drop(['price_range'],axis=1)
X_t = np.array(X_t)

print("shape of Y :"+str(y_t.shape))
print("shape of X :"+str(X_t.shape))

from sklearn.preprocessing import MinMaxScaler  #变为热编码
scaler = MinMaxScaler()
X_t = scaler.fit_transform(X_t)
print("x_t\n")
print(X_t)

X_train,X_test,Y_train,Y_test = train_test_split(X_t,y_t,test_size=.20,random_state=42)  #分割数据集
print("shape of X Train :"+str(X_train.shape))
print("shape of X Test :"+str(X_test.shape))
print("shape of Y Train :"+str(Y_train.shape))
print("shape of Y Test :"+str(Y_test.shape))

#这边取不同的C算交叉验证分数
for this_C in [1,3,5,10,20,40,60,80,100]:  #11
    clf = SVC(kernel='linear',C=this_C).fit(X_train,Y_train)
    scoretrain = clf.score(X_train,Y_train)
    scoretest  = clf.score(X_test,Y_test)
    print("Linear SVM value of C:{}, training score :{:2f} "
          ", Test Score: {:2f} \n".format(this_C,scoretrain,scoretest))

#选择C=20的时候
from sklearn.model_selection import cross_val_score,StratifiedKFold,LeaveOneOut
clf1 = SVC(kernel='linear',C=20).fit(X_train,Y_train)
scores = cross_val_score(clf1,X_train,Y_train,cv=5)
strat_scores = cross_val_score(clf1,X_train,Y_train,cv=StratifiedKFold(5,random_state=10,shuffle=True))


print("The Cross Validation Score :"+str(scores))
print("The Average Cross Validation Score :"+str(scores.mean()))
print("The Stratified Cross Validation Score :"+str(strat_scores))
print("The Average Stratified Cross Validation Score :"+str(strat_scores.mean()))

#Dummy 分类器完全忽略输入数据，scikit-learn中常用的 DummyClassifier 类型：

#most_frequent: 预测值是出现频率最高的类别
#stratified : 根据训练集中的频率分布给出随机预测
#uniform: 使用等可能概率给出随机预测
#constant: 根据用户的要求, 给出常数预测.

from sklearn.dummy import DummyClassifier

for strat in ['stratified', 'most_frequent', 'prior', 'uniform']:
    dummy_maj = DummyClassifier(strategy=strat).fit(X_train,Y_train)
    print("Train Stratergy :{} \n Score :{:.2f}".format(strat,dummy_maj.score(X_train,Y_train)))
    print("Test Stratergy :{} \n Score :{:.2f}".format(strat,dummy_maj.score(X_test,Y_test)))


#绘制数据的决策边界  14
#将数据转换为数组以进行绘图
X = np.array(df.iloc[:,[0,13]])   #装入一个数组，返回目标0-13的一个数组
print(df.iloc[:,[0,13]])  #选取其中两行进行画图
y = np.array(df['price_range'])
print("Shape of X:"+str(X.shape))
print("Shape of y:"+str(y.shape))
X = scaler.fit_transform(X)  #归一化x
print(X)

# 色图，展现train集合变为内容  16
cm_dark = ListedColormap(['#ff6060', '#8282ff','#ffaa00','#fff244','#4df9b9','#76e8fc','#3ad628'])
cm_bright = ListedColormap(['#ffafaf', '#c6c6ff','#ffaa00','#ffe2a8','#bfffe7','#c9f7ff','#9eff93'])

plt.scatter(X[:,0],X[:,1],c=y,cmap=cm_dark,s=10,label=y)
plt.show()      ###这里看到的，得到没有计算前位置图片

#这里可以看到最后结果
h = .02  # step size in the mesh
C_param = 1 # No of neighbours
for weights in ['uniform', 'distance']:
    # 们创建一个neighbors Classifier实例并拟合数据。
    clf1 = SVC(kernel='linear',C=C_param)
    clf1.fit(X, y)

    # 绘制决策边界。为此，我们将为每个对象分配一个颜色
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min()-.20, X[:, 0].max()+.20
    y_min, y_max = X[:, 1].min()-.20, X[:, 1].max()+.20
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])   # ravel to flatten the into 1D and c_ to concatenate

    # 把结果放入彩色图中
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cm_bright)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(("SVM Linear Classification (kernal = linear, Gamma = '%s')"+weights)% (C_param))

plt.show()

print("The score of the above :"+str(clf1.score(X,y)))
#结果为score 0.825  #18


# （3）线性支持向量机只有C参数,这里相当于只有一层  #19
print_line()
print("（3）线性支持向量机只有C参数LinearSVC")
from sklearn.svm import LinearSVC

for this_C in [1,3,5,10,40,60,80,100]:
    clf2 = LinearSVC(C=this_C).fit(X_train,Y_train)
    scoretrain = clf2.score(X_train,Y_train)
    scoretest  = clf2.score(X_test,Y_test)
    print("Linear SVM value of C:{}, training score :{:2f} , Test Score: {:2f} \n".format(this_C,scoretrain,scoretest))

#显然，我们用SVC得到了更好的分数，我们把核函数定义为线性的，而不是线性的SVC
#LinearSVC类基于liblinear库，它实现了线性支持向量机的优化算法。
#(1)它不支持核技巧，但它几乎与训练实例数和特征数线性扩展:其训练时间复杂度约为O(m n)。
#SVC类基于libsvm库，该库实现了一种支持内核技巧的算法。
#(1)训练时间复杂度通常在O(m2 n)和O(m3 n)之间。
#(2) LinearSVC比SVC快得多(kernel="linear")

clf2 = LinearSVC(C=10).fit(X_train,Y_train)

h = .02  # step size in the mesh
for weights in ['uniform', 'distance']:
    clf2.fit(X, y)

    # 绘制决策边界。为此，我们将为每个对象分配一个颜色
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min()-.20, X[:, 0].max()+.20
    y_min, y_max = X[:, 1].min()-.20, X[:, 1].max()+.20
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z =  clf2.predict(np.c_[xx.ravel(), yy.ravel()])   # ravel to flatten the into 1D and c_ to concatenate

    # 把结果放入彩色图中
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cm_bright)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(("linearSVM Linear Classification (kernal = linear, Gamma = '%s')"+weights)% (C_param))
plt.show()

# （4）向量机SVR和松弛因子 #20
print_line()
print("（4）向量机SVR和松弛因子")
from sklearn.svm import SVR

svr = SVR(kernel='linear',C=1,epsilon=.01).fit(X_train,Y_train)
print("{:.2f} is the accuracy of the SV Regressor".format(svr.score(X_train,Y_train)))

h = .02  # step size in the mesh
for weights in ['uniform', 'distance']:
    # 们创建一个neighbors Classifier实例并拟合数据。
    svr.fit(X, y)

    # 绘制决策边界。为此，我们将为每个对象分配一个颜色
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min()-.20, X[:, 0].max()+.20
    y_min, y_max = X[:, 1].min()-.20, X[:, 1].max()+.20
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svr.predict(np.c_[xx.ravel(), yy.ravel()])   # ravel to flatten the into 1D and c_ to concatenate

    # 把结果放入彩色图中
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cm_bright)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(("SVR Linear Classification (kernal = linear, Gamma = '%s'，epsilon=.01)"+weights)% (C_param))

plt.show()