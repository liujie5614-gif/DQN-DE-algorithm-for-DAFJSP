import pickle
import random

import matplotlib.pyplot as plt
import seaborn
# plt.switch_backend('TkAgg')

# color = ['red', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta', 'SlateBlue',
#          'Cyan', 'ghostwhite', 'goldenrod', 'navajowhite', 'navy', 'sandybrown', 'gray', 'silver', 'lime', 'tan',
#          'mediumslateblue', 'bisque','white', 'lightgreen']


f_read = open("D:\学校\学习\需要经常更新的文件\论文\Paper\DAFSP\color", 'rb')
color = pickle.load(f_read)
color.pop(0)

# random.shuffle(color)
# f_save = open('color', 'wb')
# pickle.dump(color, f_save)
# f_save.close()


plt.rcParams['mathtext.fontset'] = 'stix'
plt.xticks(fontproperties = 'Times New Roman')
plt.yticks(fontproperties = 'Times New Roman')


def draw(record, filename):
    plt.figure(figsize=(8, 4), dpi=800)
    F = len(record)
    for i in range(F):
        ax = plt.subplot(F, 1, i+1)
        ax.set(title='Factory'+str(i+1))
        plt.xlim(0, 400)
        y_ticks = ['M'+str(i+1)+str(j+1) for j in range(len(record[i])-2)]
        y_ticks.append('T')
        y_ticks.append('AM'+str(i+1))
        plt.yticks([i for i in range(len(y_ticks))], labels=y_ticks,fontproperties = 'Times New Roman')
        m = len(record[i])
        for j in range(m):
            Start_Processing = record[i][j][0]
            End_Processing = record[i][j][1]
            jobs = record[i][j][2]
            n = len(jobs)
            if j == m-2:
                for k in range(n):
                    plt.barh(j+0.2*(-1)**k-0.2, width=End_Processing[k] - Start_Processing[k],
                             height=0.4, align='edge', left=Start_Processing[k],
                             color=color[jobs[k]+10],
                             edgecolor='black', linewidth=0.4)
                    plt.text(x=Start_Processing[k] + (End_Processing[k] - Start_Processing[k]) / 2,
                             y=j+0.2*(-1)**k,
                             s='J'+str(jobs[k]+1) + '-T',
                             va='center',
                             ha='center',
                             fontsize=3)
            else:
                for k in range(n):
                    plt.barh(j, width=End_Processing[k] - Start_Processing[k],
                                             height=0.8, left=Start_Processing[k],
                                             color=color[jobs[k]+10] if j < m-1 else color[jobs[k]],
                                             edgecolor='black', linewidth=0.4)
                    plt.text(x=Start_Processing[k] + (End_Processing[k] - Start_Processing[k]) / 2,
                             y=j,
                             s='J'+str(jobs[k]+1) if j < m-1 else 'P'+str(jobs[k]+1),
                             va='center',
                             ha='center',
                             fontsize=4)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(filename, dpi=800)


# cnames = {
#     'antiquewhite': '#FAEBD7',
#     'aqua': '#00FFFF',
#     'aquamarine': '#7FFFD4',
#     'azure': '#F0FFFF',
#     'beige': '#F5F5DC',
#     'bisque': '#FFE4C4',
#     'blanchedalmond': '#FFEBCD',
#     'blue': '#0000FF',
#     'blueviolet': '#8A2BE2',
#     'brown': '#A52A2A',
#     'burlywood': '#DEB887',
#     'cadetblue': '#5F9EA0',
#     'chartreuse': '#7FFF00',
#     'chocolate': '#D2691E',
#     'coral': '#FF7F50',
#     'cornflowerblue': '#6495ED',
#     'cornsilk': '#FFF8DC',
#     'crimson': '#DC143C',
#     'cyan': '#00FFFF',
#     'darkcyan': '#008B8B',
#     'darkgoldenrod': '#B8860B',
#     'darkgray': '#A9A9A9',
#     'darkgreen': '#006400',
#     'darkkhaki': '#BDB76B',
#     'darkmagenta': '#8B008B',
#     'darkolivegreen': '#556B2F',
#     'darkorange': '#FF8C00',
#     'darkorchid': '#9932CC',
#     'darkred': '#8B0000',
#     'darksalmon': '#E9967A',
#     'darkseagreen': '#8FBC8F',
#     'darkslateblue': '#483D8B',
#     'darkslategray': '#2F4F4F',
#     'darkturquoise': '#00CED1',
#     'darkviolet': '#9400D3',
#     'deeppink': '#FF1493',
#     'deepskyblue': '#00BFFF',
#     'dimgray': '#696969',
#     'dodgerblue': '#1E90FF',
#     'firebrick': '#B22222',
#     'floralwhite': '#FFFAF0',
#     'forestgreen': '#228B22',
#     'fuchsia': '#FF00FF',
#     'gainsboro': '#DCDCDC',
#     'ghostwhite': '#F8F8FF',
#     'gold': '#FFD700',
#     'goldenrod': '#DAA520',
#     'gray': '#808080',
#     'green': '#008000',
#     'greenyellow': '#ADFF2F',
#     'honeydew': '#F0FFF0',
#     'hotpink': '#FF69B4',
#     'indianred': '#CD5C5C',
#     'indigo': '#4B0082',
#     'ivory': '#FFFFF0',
#     'khaki': '#F0E68C',
#     'lavender': '#E6E6FA',
#     'lavenderblush': '#FFF0F5',
#     'lawngreen': '#7CFC00',
#     'lemonchiffon': '#FFFACD',
#     'lightblue': '#ADD8E6',
#     'lightcoral': '#F08080',
#     'lightcyan': '#E0FFFF',
#     'lightgoldenrodyellow': '#FAFAD2',
#     'lightgreen': '#90EE90',
#     'lightgray': '#D3D3D3',
#     'lightpink': '#FFB6C1',
#     'lightsalmon': '#FFA07A',
#     'lightseagreen': '#20B2AA',
#     'lightskyblue': '#87CEFA',
#     'lightslategray': '#778899',
#     'lightsteelblue': '#B0C4DE',
#     'lightyellow': '#FFFFE0',
#     'lime': '#00FF00',
#     'limegreen': '#32CD32',
#     'linen': '#FAF0E6',
#     'magenta': '#FF00FF',
#     'maroon': '#800000',
#     'mediumaquamarine': '#66CDAA',
#     'mediumblue': '#0000CD',
#     'mediumorchid': '#BA55D3',
#     'mediumpurple': '#9370DB',
#     'mediumseagreen': '#3CB371',
#     'mediumslateblue': '#7B68EE',
#     'mediumspringgreen': '#00FA9A',
#     'mediumturquoise': '#48D1CC',
#     'mediumvioletred': '#C71585',
#     'midnightblue': '#191970',
#     'mintcream': '#F5FFFA',
#     'mistyrose': '#FFE4E1',
#     'moccasin': '#FFE4B5',
#     'navajowhite': '#FFDEAD',
#     'navy': '#000080',
#     'oldlace': '#FDF5E6',
#     'olive': '#808000',
#     'olivedrab': '#6B8E23',
#     'orange': '#FFA500',
#     'orangered': '#FF4500',
#     'orchid': '#DA70D6',
#     'palegoldenrod': '#EEE8AA',
#     'palegreen': '#98FB98',
#     'paleturquoise': '#AFEEEE',
#     'palevioletred': '#DB7093',
#     'papayawhip': '#FFEFD5',
#     'peachpuff': '#FFDAB9',
#     'peru': '#CD853F',
#     'pink': '#FFC0CB',
#     'plum': '#DDA0DD',
#     'powderblue': '#B0E0E6',
#     'purple': '#800080',
#     'red': '#FF0000',
#     'rosybrown': '#BC8F8F',
#     'royalblue': '#4169E1',
#     'saddlebrown': '#8B4513',
#     'salmon': '#FA8072',
#     'sandybrown': '#FAA460',
#     'seagreen': '#2E8B57',
#     'seashell': '#FFF5EE',
#     'sienna': '#A0522D',
#     'silver': '#C0C0C0',
#     'skyblue': '#87CEEB',
#     'slateblue': '#6A5ACD',
#     'slategray': '#708090',
#     'snow': '#FFFAFA',
#     'springgreen': '#00FF7F',
#     'steelblue': '#4682B4',
#     'tan': '#D2B48C',
#     'teal': '#008080',
#     'thistle': '#D8BFD8',
#     'tomato': '#FF6347',
#     'turquoise': '#40E0D0',
#     'violet': '#EE82EE',
#     'wheat': '#F5DEB3',
#     'white': '#FFFFFF',
#     'whitesmoke': '#F5F5F5',
#     'yellow': '#FFFF00',
#     'yellowgreen': '#9ACD32'}
# for y in cnames.values():
#     if y not in color:
#         color.append(y)
# for x in seaborn.xkcd_rgb.values():
#     if x not in color:
#         color.append(x)
# color = color[:600]



