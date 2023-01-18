import numpy as np
import codecs
from sklearn.metrics import accuracy_score

feature_dict = {"色泽": ["青绿", "乌黑", "浅白"],
                "根蒂": ["蜷缩", "稍蜷", "硬挺"],
                "敲声": ["浊响", "沉闷", "清脆"],
                "纹理": ["清晰", "稍糊", "模糊"]
                }
lable_list = ["否", "是"]
feature_list = ["色泽", "根蒂", "敲声", "纹理", "密度"]


def load_txt(path):
    ans = []
    with codecs.open(path, "r", "GBK") as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.rstrip("\r\n").split(',')
            # print(d)
            re = []
            # get index
            re.append(int(d[0]))
            re.append(feature_dict.get("色泽").index(d[1]))
            re.append(feature_dict.get("根蒂").index(d[2]))
            re.append(feature_dict.get("敲声").index(d[3]))
            re.append(feature_dict.get("纹理").index(d[4]))
            re.append(float(d[5]))

            re.append(lable_list.index(d[-1]))
            ans.append(np.array(re))
            line = f.readline()
    return np.array(ans)


class Node:
    def __init__(self, attr, label, v, split):
        # label == pi: non-leaf node
        # attr == pi: leaf node
        self.attr = attr
        self.label = label
        self.attr_v = v
        self.children = []
        self.split_v = split


def is_same_on_attr(X, attrs):  # verify if attributes are all the same
    X_a = X[:, attrs]
    target = X_a[0]
    for r in range(X_a.shape[0]):
        row = X_a[r]
        if (row != target).any():
            return False
    return True


def ent(D):
    # D is a 1d np array which actually is Y
    s = 0
    for k in set(D):
        p_k = np.sum(np.where(D == k, 1, 0)) / np.shape(D)[0]
        if p_k == 0:
            # Pklog2Pk is 0
            continue
        s += p_k * np.log2(p_k)
    return -s


def Iv(a):
    sum = 0
    for i in range(len(a)):
        sum += a[i] * np.log2(a[i])
    return -sum


def gain(X, Y, attr, discrete):
    x_attr_col = X[:, attr]
    ent_Dv = []
    weight_Dv = []

    # discrete variable
    if discrete:
        for x_v in set(x_attr_col):
            index_x_equal_v = np.where(x_attr_col == x_v)
            y_x_equal_v = Y[index_x_equal_v]
            ent_Dv.append(ent(y_x_equal_v))
            weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])

        IV = Iv(weight_Dv)
        return (ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))) / IV  # Gain-ratio
        # return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))
    # continuous variable
    else:
        x_attr_col = np.array(x_attr_col)
        x_attr_sort = np.sort(x_attr_col)
        Tmid = []
        Dv = []
        IV_list = []
        # get all possible split point for continuous variable
        for x_v in range(x_attr_sort.shape[0] - 1):
            t = (x_attr_sort[x_v] + x_attr_sort[x_v + 1]) / 2
            Tmid.append(t)
        # get split info-gain
        for i in range(len(Tmid)):
            index_x_less = np.where(x_attr_col < Tmid[i])
            index_x_more = np.where(x_attr_col > Tmid[i])
            y_less = Y[index_x_less]
            y_more = Y[index_x_more]
            Dv_ne = ent(y_less)
            Dv_po = ent(y_more)
            weight_Dv.append(np.shape(y_less)[0] / np.shape(Y)[0])
            weight_Dv.append(np.shape(y_more)[0] / np.shape(Y)[0])
            IV = Iv(weight_Dv)
            IV_list.append(IV)
            D_res = Dv_ne * (np.shape(y_less)[0] / np.shape(Y)[0]) + \
                    Dv_po * (np.shape(y_more)[0] / np.shape(Y)[0])
            Gain = ent(Y) - D_res
            Dv.append(Gain)
        # split: max gain 
        # print(max(Dv))
        # print(Tmid[Dv.index(max(Dv))])
        return max(Dv) / IV_list[Dv.index(max(Dv))], Tmid[Dv.index(max(Dv))]


def dicision_tree_init(X, Y, attrs, root, purity_cal):
    if len(set(Y)) == 1:
        root.attr = np.pi
        root.label = Y[0]
        return None

    if len(attrs) == 0 or is_same_on_attr(X, attrs):
        root.attr = np.pi
        # node's label: label with max no. in Y
        root.label = np.argmax(np.bincount(Y))
        return None

    # calculate infomation gain for each split
    purity_attrs = []
    split = -1
    for i, a in enumerate(attrs):
        # information gain for discrete variable
        if (a < 4):
            p = purity_cal(X, Y, a, True)
            purity_attrs.append(p)
        # information gain for continuous variable
        else:
            p, split = purity_cal(X, Y, 4, False)  
            purity_attrs.append(p)

    # print(purity_attrs)
    chosen_index = purity_attrs.index(max(purity_attrs))
    chosen_attr = attrs[chosen_index]

    root.attr = chosen_attr
    root.label = np.pi

    if chosen_attr < 4:
        del attrs[chosen_index]

        x_attr_col = X[:, chosen_attr]
        # discrete variable
        for x_v in set(X[:, chosen_attr]):
            n = Node(-1, -1, x_v, -1)
            root.children.append(n)

            index_x_equal_v = np.where(x_attr_col == x_v)
            X_x_equal_v = X[index_x_equal_v]
            Y_x_equal_v = Y[index_x_equal_v]
            dicision_tree_init(X_x_equal_v, Y_x_equal_v, attrs, n, purity_cal)
    else:
        x_attr_col = X[:, chosen_attr]
        n1 = Node(-1, -1, 0, split)  # x < split
        n2 = Node(-1, -1, 1, split)  # x > split
        root.children.append(n1)
        root.children.append(n2)
        index_x_less_v = np.where(x_attr_col < split)
        X_x_less_v = X[index_x_less_v]
        Y_x_less_v = Y[index_x_less_v]
        dicision_tree_init(X_x_less_v, Y_x_less_v, attrs, n1, purity_cal)
        index_x_more_v = np.where(x_attr_col > split)
        X_x_more_v = X[index_x_more_v]
        Y_x_more_v = Y[index_x_more_v]
        dicision_tree_init(X_x_more_v, Y_x_more_v, attrs, n2, purity_cal)


def dicision_tree_predict(x, tree_root):
    # leaf node
    if tree_root.label != np.pi:
        return tree_root.label

    # make decision
    if tree_root.label == np.pi and tree_root.attr == np.pi:
        print("err!")
        return None

    chose_attr = tree_root.attr
    # choose branch
    if chose_attr == 4:
        if x[chose_attr] <= tree_root.split_v:
            x[chose_attr] = 0
        else:
            x[chose_attr] = 1
    for child in tree_root.children:
        if child.attr_v == x[chose_attr]:
            return dicision_tree_predict(x, child)

    return None


if __name__ == '__main__':
    ans = load_txt("Watermelon-train2.csv")
    X_train = ans[:, 1: -1]
    Y_train = ans[:, -1]
    Y_train.astype(np.int64)
    # print(X_train)
    # print(Y_train)

    test_data = load_txt("Watermelon-test2.csv")
    X_test = test_data[:, 1:-1]
    Y_test = test_data[:, -1]

    r = Node(-1, -1, -1, -1)
    attrs = [0, 1, 2, 3, 4]  

    dicision_tree_init(X_train, Y_train, attrs, r, gain)

    y_predict = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        y_p = dicision_tree_predict(x, r)
        y_predict.append(y_p)

    acc = accuracy_score(Y_test, y_predict)
    print('accuracy:', acc)
