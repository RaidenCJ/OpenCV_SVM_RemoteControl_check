# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''


'''

import os
import sys
import shutil

    
def copy_files(path, correct_path, incorrect_path):
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                pos = line.find(" ")
                file_path = line[:pos]
                print(file_path)
                new_file_path = ""
                file_name = os.path.basename(file_path)
                print(file_name)

                if "0 0" in line:
                    new_file_path = os.path.join(incorrect_path, file_name)
                elif "0 1" in line:
                    new_file_path = os.path.join(correct_path, file_name)
                print(new_file_path)
                shutil.copyfile(file_path, new_file_path)
                #if os.path.exists(new_file_path):
                #    shutil.copyfile(file_path, new_file_path)                                
    else:
        print(path + " is not exist")


copy_files(sys.argv[1], sys.argv[2], sys.argv[3])



