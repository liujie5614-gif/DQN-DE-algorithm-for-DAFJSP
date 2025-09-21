import pickle
import ENV
from Train_DE import train_pop
from train_individual import train_in

jobnums = [_ for _ in range(20, 101, 5)]
pnums = [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10]
fnums = [2, 3, 4, 5]

for i in range(len(jobnums)):
    print('start training: instance' + str(jobnums[i]) + '_' + str(pnums[i]))
    Jnum, Onum, Pnum, Fnum, Mnum = jobnums[i], 4, pnums[i], 2, 6
    train_pop(Jnum, Onum, Pnum, Fnum, Mnum)
    print('finish training: instance' + str(jobnums[i]) + '_' + str(pnums[i]))
