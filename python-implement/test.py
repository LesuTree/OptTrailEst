import math

import present

s_box = present.s_box
p_estim = 0.00020
p_Best = []
Round = 2
p_layer_order = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
prodRound = []
prodRoundDetail = []
alpha_result = []
beta_result = []

alpha_characteristic = []
beta_characteristic = []
# 示例
blocksize = 16
n = 2  # 指定要遍历的分量个数
alpha_result = []
beta_result = []


def s_box_diff_distribution(s_box):
    size = len(s_box)
    diff_distribution = {}

    for in_diff in range(size):
        for input1 in range(size):
            output1 = s_box[input1]
            input2 = input1 ^ in_diff
            output2 = s_box[input2]
            out_diff = output1 ^ output2

            if (in_diff, out_diff) in diff_distribution:
                diff_distribution[(in_diff, out_diff)] += 1
            else:
                diff_distribution[(in_diff, out_diff)] = 1

    return diff_distribution


# 例子
s_box = present.s_box
result = s_box_diff_distribution(s_box)

# 输出差分分布
for diff_pair, count in result.items():
    in_diff, out_diff = diff_pair
    print(f"Input Difference: {hex(in_diff)}, Output Difference: {hex(out_diff)}, Count: {count}")
from itertools import product


def find_max_diff_input_for_output(diff_output, diff_distribution):
    max_count = 0
    max_diff_input = None

    for diff_input, count in diff_distribution.items():
        if diff_input[1] == diff_output and count > max_count:
            max_count = count
            max_diff_input = diff_input[0]

    return max_diff_input, max_count


diff_output_to_find = 0x0
max_diff_input, max_count = find_max_diff_input_for_output(diff_output_to_find, result)
print(f"For Diff Output {hex(diff_output_to_find)}, Max Diff Input: {hex(max_diff_input)}, Max Count: {max_count}")


def generate_permutations(blocksize):
    vector_length = blocksize // 4
    value_range = range(16)  # 可调整为实际的取值范围

    # 生成所有可能的组合
    permutations = list(product(value_range, repeat=vector_length))

    print(len(permutations))
    value_range = [permutations[0][i] for i in range(0, 4)]
    print(value_range[0])
    max_diff_input, max_count = find_max_diff_input_for_output(value_range[0], result)
    print(result)
    print(hex(max_diff_input))
    print(max_count)


# 示例
blocksize = 16

generate_permutations(blocksize)

from itertools import combinations, product


def find_max_diff_input_excluding_zero(diff_distribution):
    max_count = 0
    max_diff_input = None

    for diff_input, count in diff_distribution.items():
        if diff_input[0] != 0 and count > max_count:
            max_count = count
            max_diff_input = diff_input[0]

    return max_diff_input, max_count


def pSB(active_sbox, diff_distribution):
    max_diff_input_excluding_zero, max_count_excluding_zero = find_max_diff_input_excluding_zero(diff_distribution)
    prob = float(max_count_excluding_zero) / 16
    return prob ** active_sbox


p_Best.append(pSB(1, result))
print(f"p_Best[0]:{p_Best[0]}")


def rankRound(rnd):
    print(f"loat(p_estim) {p_estim}")
    print(f"p_Best[Round - rnd-1]{p_Best[Round - rnd - 1]}")
    return float(p_estim) / float(p_Best[Round - rnd - 1])


# # 基于beta_0进行换算，输入的格式为['0000', '0000', '1111', '1110']
# def searchSubRound(rnd, param,active_Sbox):
#     if param > active_Sbox:
#         prodRound.append(math.prod(prodRoundDetail[rnd-1]))
#     for beta_index in beta_result[rnd-1]:

def decimal_array_to_binary_string(decimal_array):
    binary_string = ""
    for decimal_number in decimal_array:
        # 将十进制数字转换为4位的二进制字符串
        binary_string += format(decimal_number, '04b')

    return binary_string


vectors_test = [0, 13, 15, 0]
bits = decimal_array_to_binary_string(vectors_test)
print('bits', bits)


def sort_by_distribution(input_diff):
    # 使用 sorted 函数和 lambda 函数按照次数从大到小排序
    sorted_result = sorted(
        [(output_diff, count) for (in_diff, output_diff), count in result.items() if in_diff == input_diff],
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_result


# 示例调用
input_diff = 15
sorted_output = sort_by_distribution(input_diff)

# 输出排序结果
print(f"输入差分 {input_diff} 对应的输出差分和分布次数：")
for output_diff, count in sorted_output:
    print(f"输出差分: {output_diff}, 分布次数: {count}")


def get_decreasing_beta(alpha_value):
    pass


def getList(list):
    list_values = [0] * len(list)
    for i in range(len(list)):
        list_values[i] = 0 if list[i] == 0 else 1
    return list_values


def getL(list, index):
    list_values = getList(list)
    for i in range(0, index + 1):
        list_values[i] = 0
    return list_values


def getProd(L_r_n):
    curr = 1
    for i in range(len(L_r_n)):
        if L_r_n[i] != 0:
            curr *= 0.25
    return curr


def clearAll():
    alpha_result.clear()
    beta_result.clear()
    prodRound.clear()
    prodRoundDetail.clear()


def searchSubRound(rnd, active_sbox_curr, active_sbox, active_index):
    if active_sbox_curr > active_sbox:
        prodRound[rnd - 1] = math.prod(prodRoundDetail[rnd - 1])
        beta_rnd_1 = beta_result[rnd]
        beta_rnd_1_bits = decimal_array_to_binary_string(beta_rnd_1)
        state_p_layer = [0 for _ in range(blocksize)]
        for p_index, std_bits in enumerate(beta_rnd_1_bits):
            state_p_layer[p_layer_order[p_index]] = std_bits
        alpha_rnd_1_bits = state_p_layer
        alpha_rnd_1 = [int(''.join(alpha_rnd_1_bits[i:i + 4]), 2) for i in range(0, len(alpha_rnd_1_bits), 4)]
        global alpha_result
        alpha_result.append(alpha_rnd_1)

        if rnd + 1 < Round:
            searchRound(rnd + 1)
        else:
            lastRound()
    else:
        # For each brρr(n) sorted in decreasing order according to Pρr(n)(arρr(n) → ·)
        input_diff = 15
        sorted_beta_output = sort_by_distribution(alpha_result[rnd - 1][active_index[active_sbox_curr]])
        for beta_output, beta_count in sorted_beta_output:
            beta_result[rnd-1][active_index[active_sbox_curr]] = beta_output
            L_r_n = getL(alpha_result[rnd - 1], active_index[active_sbox_curr])
            prod_L_r_n = getProd(L_r_n)
            prodRoundDetail[rnd - 1][active_index[active_sbox_curr]] = float(beta_output) / 16

            prod = math.prod(prodRound[0:rnd - 2]) * math.prod(prodRoundDetail[0:active_index[active_sbox_curr]]) * prod_L_r_n
            if prod < rankRound(rnd):
                clearAll()
                break
            beta_result_rnd = beta_result[rnd-1]
            beta_result_bits = decimal_array_to_binary_string(beta_result_rnd)
            state_p_layer_2 = [0 for _ in range(blocksize)]
            for p_index, std_bits in enumerate(beta_result_bits):
                state_p_layer_2[p_layer_order[p_index]] = std_bits
            alpha_result_bits = state_p_layer_2
            alpha_result_new = [int(''.join(alpha_result_bits[i:i + 4]), 2) for i in range(0, len(alpha_result_bits), 4)]

            alpha_result_list = getList(alpha_result_new)
            prod_alpha_result = getProd(alpha_result_list)
            if prod * prod_alpha_result >= rankRound(rnd+1):
                searchSubRound(rnd,active_sbox_curr+1,active_sbox,active_index)
            # Lr, n ←list(αr)∧(0ρr(n)1N−ρr(n))
            # 还差最后三行！！！

        # 这里实现从1-active_sbox的连乘


def searchRound(rnd):
    active_sbox = 0
    active_index = []
    # 计算当前轮的alpha中有多少个非零值和对应的下标
    for index, alpha_rnd_index in enumerate(alpha_result[rnd - 1]):
        if alpha_rnd_index != '0':
            active_sbox += 1
            active_index.append(index)

    searchSubRound(rnd, 1, active_sbox, active_index)


def find_max_output_count_for_input(diff_input, diff_distribution):
    max_count = 0
    max_diff_output = None

    for diff_output, count in diff_distribution.items():
        if diff_output[0] == diff_input and count > max_count:
            max_count = count
            max_diff_output = diff_output[1]

    return max_diff_output, max_count


def saveCharacteristic(prob):
    print("we are saving")
    global alpha_characteristic
    global beta_characteristic
    global p_estim
    alpha_characteristic = [alpha_result_index for alpha_result_index in alpha_result]
    beta_characteristic = [beta_result_index for beta_result_index in beta_result]
    p_estim = prob


def lastRound():
    beta_r = []
    alpha_r_count = []
    for alpha_r_index in alpha_result[len(alpha_result) - 1]:
        max_diff_output_r_index, max_diff_output_r_count = find_max_output_count_for_input(alpha_r_index, result)
        beta_r.append(max_diff_output_r_index)
        alpha_r_count.append(max_diff_output_r_count)
    print(f"beta{beta_r}")
    print(f"beta_count{alpha_r_count}")
    beta_result.append(beta_r)
    pd_r = 1
    for count_index in alpha_r_count:
        pd_r *= float(count_index) / 16
    prodRoundDetail.append(alpha_r_count)
    prodRound.append(pd_r)
    current_pro = math.prod(prodRound)
    print(f"pd{prodRound}")
    print(f"current_pro{current_pro}")
    if current_pro >= p_estim:
        saveCharacteristic(current_pro)
    prodRound.clear()
    alpha_result.clear()
    beta_result.clear()


def firstRound(beta_0_vector):
    alpha_0 = []
    alpha_0_count = []
    for beta_0_index, beta_0_value in enumerate(beta_0_vector):
        max_beta_0_index, max_beta_0_count = find_max_diff_input_for_output(int(beta_0_value, 2), result)
        alpha_0.append(max_beta_0_index)
        alpha_0_count.append(max_beta_0_count)
    alpha_result.append(alpha_0)
    beta_0 = [int(beta_0_vector[i], 2) for i in range(0, len(beta_0_vector))]
    beta_result.append(beta_0)
    # 遍历count后，进行相乘
    pd_0 = 1
    for count in alpha_0_count:
        pd_0 *= float(count) / 16
    prodRoundDetail.append(alpha_0_count)
    prodRound.append(pd_0)

    # 进行迭代
    beta_0_bits = ''.join(beta_0_vector)
    state_p_layer = [0 for _ in range(blocksize)]
    for p_index, std_bits in enumerate(beta_0_bits):
        state_p_layer[p_layer_order[p_index]] = std_bits
    alpha_1_bits = state_p_layer
    alpha_1 = [int(''.join(alpha_1_bits[i:i + 4]), 2) for i in range(0, len(alpha_1_bits), 4)]
    alpha_result.append(alpha_1)
    print(f"alpha_1{alpha_1}")
    print(f"alpha_0{alpha_0}")
    # 调用round
    if Round > 2:
        searchRound(2)
    else:
        lastRound()

    return alpha_0, alpha_0_count


# alpha_0_test,alpha_0_count_test = firstRound(['0000', '0000', '1111', '1110'])
# print(f"alpha_0_test{alpha_0_test}")
# print(f"alpha_0_count_testt{alpha_0_count_test}")
# print(alpha_characteristic)
# print(beta_characteristic)
# print(p_estim)
def traverse_partial_vectors(blocksize, alpha, beta):
    vector_length = blocksize // 4
    value_range = range(16)  # 可调整为实际的取值范围
    print(vector_length)
    for active_sbox in range(1, vector_length + 1):
        # TODO 添加一个分支判定的条件
        print(f"rankOund[1]{rankRound(1)}")
        print(f"pSB:{pSB(active_sbox, result)}")
        if pSB(active_sbox, result) < rankRound(1):
            print("舍弃！")
            print(active_sbox)
            break

        # 生成所有可能的组合，其中n个分量进行遍历
        combination_indices = list(combinations(range(vector_length), active_sbox))

        print('---------------', len(combination_indices))
        for indices_combination in combination_indices:
            partial_vector_values = product(value_range, repeat=active_sbox)
            # 使用 filter 过滤掉包含值为0的 partial_values
            filtered_partial_vectors = filter(lambda partial_values: 0 not in partial_values, partial_vector_values)

            for partial_values in filtered_partial_vectors:
                full_vector = [0] * vector_length  # 初始化为全0向量
                print(f"indices_combination{indices_combination}")

                for i, value in zip(indices_combination, partial_values):
                    full_vector[i] = value
                # print(f"Full Vector: {full_vector}")
                beta_0_vector = [bin(value)[2:].zfill(4) for value in full_vector]
                # print(f"binary_vector{beta_0_vector}")
                beta_0_string = ''.join(beta_0_vector)
                # TODO 赋值之后，要进行调用firstRound
                alpha_0_test, alpha_0_count_test = firstRound(beta_0_vector)
                print(f"alpha_0_test{alpha_0_test}")
                print(f"alpha_0_count_testt{alpha_0_count_test}")


# 示例
blocksize = 16
n = 2  # 指定要遍历的分量个数
alpha_result = []
beta_result = []
traverse_partial_vectors(blocksize, alpha_result, beta_result)
print(alpha_characteristic)
print(beta_characteristic)
print(f"p_estim{p_estim}")

# 示例
diff_distribution = {
    (0x0, 0x2): 5,
    (0x1, 0x2): 8,
    (0x2, 0x2): 12,
    # Add more entries as needed
}

max_diff_input_excluding_zero, max_count_excluding_zero = find_max_diff_input_excluding_zero(diff_distribution)
print(
    f"For Nonzero Input Diff, Max Diff Input: {hex(max_diff_input_excluding_zero)}, Max Count: {max_count_excluding_zero}")

max_prob = pSB(2, result)
print(f"max_prob_2:{max_prob}")

decimal_vector = [0, 13, 15, 0]

binary_vector = [bin(value)[2:].zfill(4) for value in decimal_vector]
binary_string = ''.join(binary_vector)

print("Decimal Vector:", decimal_vector)
print("Binary Vector:", binary_vector)
print("Binary String (length 16):", binary_string)
