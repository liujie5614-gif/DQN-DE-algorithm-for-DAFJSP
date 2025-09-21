import pickle

import xlwt

import ENV

Jnum, Onum, Pnum, Fnum, Mnum = 60, 4, 6, 2, 6
schedule_instance_name = 'D:\\学校\\学习\\python\\Paper\\DAFSP\\data\\' + str(Jnum) + '_' + str(Pnum) + '.pkl'
# f_read = open(schedule_instance_name, 'rb')
# schedule_instance = pickle.load(f_read)
# index = 0
# for i in range(len(schedule_instance['Products'])):
#     for j in range(len(schedule_instance['Products'][i])):
#         schedule_instance['Products'][i][j] = index
#         index += 1
# f_save = open(schedule_instance_name, 'wb')
# pickle.dump(schedule_instance, f_save)
# f_save.close()

f_read = open(schedule_instance_name, 'rb')
schedule_instance = pickle.load(f_read)
data1 = schedule_instance['jobT']
data2 = schedule_instance['Machine']
data3 = schedule_instance['Products']
data4 = schedule_instance['AssembleTime']
data5 = schedule_instance['Tansporttime']
wb = xlwt.Workbook()
sh1 = wb.add_sheet('时间')
sh2 = wb.add_sheet('机器')
sh3 = wb.add_sheet('产品')
sh4 = wb.add_sheet('装配时间')
sh5 = wb.add_sheet('运输时间')

for i in range(len(data1)):
    for j in range(len(data1[i])):
        s1, s2, s3, s4, s5 = '', '', '', '', ''
        a = data1[i][j]
        for k in range(len(data1[i][j])):
            if k == len(data1[i][j])-1:
                s1 = s1 + str(data1[i][j][k])
                s2 = s2 + str(data2[i][j][k]+1)
            else:
                s1 = s1 + str(data1[i][j][k]) + ','
                s2 = s2 + str(data2[i][j][k]+1) + ','
        sh1.write(i, j, s1)
        sh2.write(i, j, s2)
for i in range(len(data3)):
    s3 = ''
    s5 = ''
    for j in range(len(data3[i])):
        if j == len(data3[i]) - 1:
            s3 += str(data3[i][j])
            s5 += str(data5[i][j])
        else:
            s3 += str(data3[i][j]) + ','
            s5 += str(data5[i][j]) + ','
    s4 = str(data4[i])
    sh5.write(i, 1, s5)
    sh4.write(i, 1, s4)
    sh3.write(i, 1, s3)
wb.save('real-life-instance.xls')

