# 更新 2025/4/10
增加对最终输出的蒸馏，即196*feature tokens + 1*cls tokens的蒸馏，命名为out_loss
相应地loss_weight变为0 0 0 0 0 1，见all.sh

# 更新 2025/4/17
增加时间记录，将总时间输出到log.txt中
增加超参数--loss，可以选择cosine loss作为pretrain损失