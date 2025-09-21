import numpy as np
import xlwt
from DQN_DE import Solve_DAFJSP_DQN_DE
from DE_main import Solve_DAFJSP_DE
from IGA import Solve_DAFJSP_IGA
from HABC import Solve_DAFJSP_HABC

wb = xlwt.Workbook()
sh1 = wb.add_sheet('avg')
sh2 = wb.add_sheet('min')
sh3 = wb.add_sheet('std')
sh1.write(0, 0, 'instance')
sh1.write(0, 1, 'DQN-DE')
sh1.write(0, 2, 'DE')
sh1.write(0, 3, 'IGA')
sh1.write(0, 4, 'HABC')
jobnums = [_ for _ in range(15, 101, 5)]
pnums = [5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10]
# fnums_s = [2, 3]
# fnums_l = [2, 3, 4, 5]
fnums = [2, 3, 4, 5]
row_index = 1

try:
    for i in range(len(jobnums)):
        if jobnums[i] <= 30:
            fnum = 2
        else:
            fnum = 4
        for j in range(fnum):
            Jnum, Onum, Pnum, Fnum, Mnum = jobnums[i], 4, pnums[i], fnums[j], 6
            sh1.write(row_index, 0, str(Jnum) + '_' + str(Pnum) + '_' + str(Fnum))
            sh2.write(row_index, 0, str(Jnum) + '_' + str(Pnum) + '_' + str(Fnum))
            sh3.write(row_index, 0, str(Jnum) + '_' + str(Pnum) + '_' + str(Fnum))
            results_DQN_DE = []
            results_DE = []
            results_IGA = []
            results_HABC = []
            for k in range(10):
                best_DQN_DE, _, _ = Solve_DAFJSP_DQN_DE(Jnum, Onum, Pnum, Fnum, Mnum)
                results_DQN_DE.append(best_DQN_DE[0])
                best_DE, _ = Solve_DAFJSP_DE(Jnum, Onum, Pnum, Fnum, Mnum)
                results_DE.append(best_DE[0])
                best_IGA, _ = Solve_DAFJSP_IGA(Jnum, Onum, Pnum, Fnum, Mnum)
                results_IGA.append(best_IGA[0])
                best_HABC, _ = Solve_DAFJSP_HABC(Jnum, Onum, Pnum, Fnum, Mnum)
                results_HABC.append(best_HABC[0])

            avg_DQN_DE = np.mean(results_DQN_DE)
            avg_DE = np.mean(results_DE)
            avg_IGA = np.mean(results_IGA)
            avg_HABC = np.mean(results_HABC)
            sh1.write(row_index, 1, float(avg_DQN_DE))
            sh1.write(row_index, 2, float(avg_DE))
            sh1.write(row_index, 3, float(avg_IGA))
            sh1.write(row_index, 4, float(avg_HABC))

            min_DQN_DE = np.min(results_DQN_DE)
            min_DE = np.min(results_DE)
            min_IGA = np.min(results_IGA)
            min_HABC = np.min(results_HABC)
            sh2.write(row_index, 1, float(min_DQN_DE))
            sh2.write(row_index, 2, float(min_DE))
            sh2.write(row_index, 3, float(min_IGA))
            sh2.write(row_index, 4, float(min_HABC))

            std_DQN_DE = np.std(results_DQN_DE)
            std_DE = np.std(results_DE)
            std_IGA = np.std(results_IGA)
            std_HABC = np.std(results_HABC)
            sh3.write(row_index, 1, float(std_DQN_DE))
            sh3.write(row_index, 2, float(std_DE))
            sh3.write(row_index, 3, float(std_IGA))
            sh3.write(row_index, 4, float(std_HABC))
            row_index += 1
except Exception as e:
    print(e)
    wb.save('Comparison_experiment.xls')
finally:
    wb.save('Comparison_experiment.xls')

