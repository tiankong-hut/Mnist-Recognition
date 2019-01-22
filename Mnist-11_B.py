# coding=utf-8
# 定义分类函数
# 只调用单个函数:
# from B import C
# if __name__ == "__main__":
#     C(x,y)

# 同一目录下:
# import Mnist-11_B
# if __name__ == "__main__":
#     Mnist-11_B.classify()


def classify():

    #初始化参数
    A0_to_0 = 0
    A0_to_other = 0
    other_to_0 = 0

    A4_to_4 = 0
    A4_to_other = 0
    other_to_4 = 0

    for i in range(0, 100):  # for循环：对每个元素都执行相同的操作
        # print "i=: ",i
        if arg_pre[i] == arg_y[i]:
            print "正确识别的数字:-- ", arg_pre[i]

        if arg_y[i] == arg_pre[i] == 0:
            A0_to_0 = A0_to_0 + 1
            # print "正确的数字: ", arg_pre[i]
            # print "A0_to_0--正确的个数: ", A0_to_0
        elif arg_y[i] != 0 and arg_pre[i] == 0:
            other_to_0 += 1
            # print "other_to_0--误识的个数: ", other_to_0
        elif arg_y[i] == 0 and arg_pre[i] != 0:
            A0_to_other += 1
            # print "A0_to_other--拒识的个数: ", A0_to_other

        # elif  arg_pre[i] == arg_y[i] == 1:
        #     print "正确的数: ", arg_pre[i]
        # elif  arg_pre[i] == arg_y[i] == 2:
        # elif  arg_pre[i] == arg_y[i] == 3:
        if arg_pre[i] == arg_y[i] == 4:
            A4_to_4 = A4_to_4 + 1
            # print "A4_to_4--正确的个数: ", A4_to_4
        elif arg_y[i] != 4 and arg_pre[i] == 4:
            other_to_0 += 1
            # print "other_to_4--误识的个数: ", other_to_4
        elif arg_y[i] == 4 and arg_pre[i] != 4:
            A4_to_other += 1
            # print "A4_to_other--拒识的个数: ", A4_to_other






    print "A0_to_0--正确识别的个数:  ", A0_to_0
    print "other_to_0--误识的个数:  ", other_to_0
    print "A0_to_other--拒识的个数:  ", A0_to_other

    print "A1_to_1--正确识别的个数: ", A1_to_1
    print "other_to_1--误识的个数: ", other_to_1
    print "A1_to_other--拒识的个数: ", A1_to_other

    print "A2_to_2--正确识别的个数: ", A2_to_2
    print "other_to_2--误识的个数: ", other_to_2
    print "A2_to_other--拒识的个数:  ", A2_to_other

    print "A3_to_3--正确识别的个数: ", A3_to_3
    print "other_to_3--误识的个数: ", other_to_3
    print "A3_to_other--拒识的个数: ", A3_to_other

    print "A4_to_4--正确识别的个数: " , A4_to_4
    print "other_to_4--误识的个数:  " , other_to_4
    print "A4_to_other--拒识的个数: " , A4_to_other
    # print "%.3f" % (A4_to_4 / A4)

    print "A5_to_5--正确识别的个数: ", A5_to_5
    print "other_to_5--误识的个数: ", other_to_5
    print "A5_to_other--拒识的个数: ", A5_to_other

    print "A6_to_6--正确识别的个数: ", A6_to_6
    print "other_to_6--误识的个数: ", other_to_6
    print "A6_to_other--拒识的个数: ", A6_to_other

    print "A7_to_7--正确识别的个数: ", A7_to_7
    print "other_to_7--误识的个数: ", other_to_7
    print "A7_to_other--拒识的个数: ", A7_to_other

    print "A8_to_8--正确识别的个数: ", A8_to_8
    print "other_to_8--误识的个数: ", other_to_8
    print "A8_to_other--拒识的个数: ", A8_to_other

    print "A9_to_9--正确识别的个数: ", A9_to_9
    print "other_to_9--误识的个数: ", other_to_9
    print "A9_to_other--拒识的个数: ", A9_to_other