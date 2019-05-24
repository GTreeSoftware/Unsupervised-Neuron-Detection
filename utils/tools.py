import os
import shutil
import csv

def make_dirs(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)

def remake_dirs(pathname):
    if os.path.exists(pathname):
        shutil.rmtree(pathname)
    os.makedirs(pathname)

def load_file():
    # 获取当前文件路径
    current_path = os.path.abspath(__file__)
    # 获取当前文件的父目录
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # config.ini文件路径,获取当前目录的父目录的父目录与congig.ini拼接
    config_file_path=os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),'config.ini')
    print('当前目录:' + current_path)
    print('当前父目录:' + father_path)
    print('config.ini路径:' + config_file_path)


# wirte the csv files
def list2csv(list, file, mode='a+'):
    with open(file, mode) as f:
        w=csv.writer(f)
        w.writerow(list)



def csv_write(out_filename, in_header_list, in_val_list):
    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(in_header_list)
        writer.writerows(zip(*in_val_list))


