# 更新
增加对最终输出的蒸馏，即196*feature tokens + 1*cls tokens的蒸馏，命名为out_loss
相应地loss_weight变为0 0 0 0 0 1，见all.sh
