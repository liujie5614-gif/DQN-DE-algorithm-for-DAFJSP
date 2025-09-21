import pickle
import ENV
from Train_DE import train_pop
from train_individual import train_in

# jobnums = [_ for _ in range(20, 101, 5)]
# pnums = [5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10]
jobnums = [_ for _ in range(85, 101, 5)]
pnums = [10, 10, 10, 10]
fnums = [2, 3, 4, 5]

for i in range(len(jobnums)):
    print('start training: instance' + str(jobnums[i]) + '_' + str(pnums[i]))
    Jnum, Onum, Pnum, Fnum, Mnum = jobnums[i], 4, pnums[i], 2, 6
    train_in(Jnum, Onum, Pnum, Fnum, Mnum)
    print('finish training: instance' + str(jobnums[i]) + '_' + str(pnums[i]))

# # 生成实例并保存
# def generateins():
#     for i in range(len(jobnums)):
#         Jnum, Onum, Pnum, Fnum, Mnum = jobnums[i], 4, pnums[i], 2, 6
#         instance = ENV.Schedule_Instance(Jnum, Onum, Pnum, Fnum, Mnum)
#         schedule_instance = {'jobT': instance.jobT, 'Products': instance.Products, 'Machine': instance.Machine,
#                              'AssembleTime': instance.AssembleTime, 'Tansporttime': instance.Tansporttime}
#         schedule_instance_name = 'D:\\学校\\学习\\python\\Paper\\DAFSP\\data\\' + str(Jnum) + '_' + str(Pnum) + '.pkl'
#         f_save = open(schedule_instance_name, 'wb')
#         pickle.dump(schedule_instance, f_save)
#         f_save.close()
#
#
# generateins()