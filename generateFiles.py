# -*- coding: UTF-8 -*-


#文本格式
#路径+文件名+空格+标签

import os


SVM_FILE_TRAIN = "/home/zhgg/share/workspace/svm/total_svm_data_train.txt"
CORRECT_IMG_PATH_TRAIN = "/home/zhgg/share/workspace/svm/train/total/with_remote"
INCORRECT_IMG_PATH_TRAIN = "/home/zhgg/share/workspace/svm/train/total/no_remote"

SVM_FILE_TEST = "/home/zhgg/share/workspace/svm/total_svm_data_test.txt"
CORRECT_IMG_PATH_TEST = "/home/zhgg/share/workspace/svm/test/total/with_remote"
INCORRECT_IMG_PATH_TEST = "/home/zhgg/share/workspace/svm/test/total/no_remote"



def DeleteFile(myfile):
	if os.path.exists(myfile):
		os.remove(myfile)

def GenerateFile(imgPath, filePath, label):
	if os.path.exists(imgPath):
		file = open(filePath, 'a+')
		for (root, dirs, files) in os.walk(imgPath):
			for filename in files:
				pathName = os.path.join(root, filename)
				file.write(pathName + " " + label + "\n")
	else:
		print (imgPath + " is not exist")


DeleteFile(SVM_FILE_TRAIN)		
GenerateFile(CORRECT_IMG_PATH_TRAIN, SVM_FILE_TRAIN, "1")
GenerateFile(INCORRECT_IMG_PATH_TRAIN, SVM_FILE_TRAIN, "0")

DeleteFile(SVM_FILE_TEST)		
GenerateFile(CORRECT_IMG_PATH_TEST, SVM_FILE_TEST, "1")
GenerateFile(INCORRECT_IMG_PATH_TEST, SVM_FILE_TEST, "0")
