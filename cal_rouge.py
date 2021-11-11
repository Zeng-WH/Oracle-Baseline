# prefix计算rouge分数

from rouge import Rouge
f = open("/home/ypd-19-2/prefix-tuning/QMSum-transfer/product_as_test/lead/train.oracle", "r", encoding="utf-8")
f1 = open("/home/ypd-19-2/prefix-tuning/QMSum-transfer/product_as_test/lead/train.tgt", "r", encoding="utf-8")
cands = []
golds = []
f_lines = f.readlines()
f1_lines = f1.readlines()
#c = f.readlines()
for line in f_lines:
    cand = line.strip().replace("<q>", " ")
    cands.append(cand)
for line in f1_lines:
    gold = line.strip().replace("<q>", " ")
    golds.append(gold)

rouge = Rouge()
total_R1_P = 0
total_R1_R = 0
total_R1_F = 0
total_R2_P = 0
total_R2_R = 0
total_R2_F = 0
total_RL_P = 0
total_RL_R = 0
total_RL_F = 0
for i in range(len(cands)):
    if cands[i] == '':
        cands[i] = 'the user'
    if golds[i] == '':
        golds[i] = 'the user'
    rouge_score = rouge.get_scores(cands[i], golds[i])
    R_1 = rouge_score[0]["rouge-1"]
    R_2 = rouge_score[0]["rouge-2"]
    R_L = rouge_score[0]["rouge-l"]
    P_R_1 = R_1['p']
    R_R_1 = R_1['r']
    F_R_1 = R_1['f']
    P_R_2 = R_2['p']
    R_R_2 = R_2['r']
    F_R_2 = R_2['f']
    P_R_L = R_L['p']
    R_R_L = R_L['r']
    F_R_L = R_L['f']
    total_R1_P += P_R_1
    total_R1_R += R_R_1
    total_R1_F += F_R_1
    total_R2_P += P_R_2
    total_R2_R += R_R_2
    total_R2_F += F_R_2
    total_RL_P += P_R_L
    total_RL_R += R_R_L
    total_RL_F += F_R_L
print("R1 Score:")
print(total_R1_P/len(cands))
print(total_R1_R/len(cands))
print(total_R1_F/len(cands))
print("R2 Score:")
print(total_R2_P/len(cands))
print(total_R2_R/len(cands))
print(total_R2_F/len(cands))
print("RL Score:")
print(total_RL_P/len(cands))
print(total_RL_R/len(cands))
print(total_RL_F/len(cands))
