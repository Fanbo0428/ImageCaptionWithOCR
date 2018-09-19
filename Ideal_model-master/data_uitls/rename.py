# -*- coding:utf-8 -*-
import os 

class ImageRename():
    def __init__(self):
        self.path = '/Users/apple/Desktop/raw_data/img'
    
    def rename(self):
        file_list = os.listdir(self.path)
        total_num = len(file_list)
        i = 0
        for item in file_list:
            if item.endswith('.jpeg') or item.endswith('.jpg') or item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '000' + format(str(i), '0>3s') + '.jpg')
                os.rename(src, dst)
                print 'converting %s to %s ...' % (src, dst)
                i+=1
        print 'total %d to rename & converted %d jpgs' % (total_num, i)
        
if __name__ == '__main__':
    newname=ImageRename()
    newname.rename()