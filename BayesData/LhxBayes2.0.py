#!/usr/bin/env python
# -*- coding:utf-8 -*-
# This Python codes are created by Li hexi for undergraduate Machine learning course
# Any copy of the codes is illegal without the author' permission.2021-5-3
import math
import cv2
from matplotlib import pyplot as plt
import tkinter as tk
import tkinter.filedialog as file
import tkinter.messagebox as mes
import jieba
import numpy as np
import docx
from tkinter import scrolledtext as st
# GUI 初始化
MainWin = tk.Tk()
MainWin.title('朴素贝叶斯学习算法')
MainWin.geometry('1200x640')
MainWin.resizable(width=False, height=False)
TagFlag=False#打标注标志
TrainFlag=False#训练标志
FinishFlag=False#完成标志
ClassNum=3#文档分类的类别数
words=[]#临时打开的文档，还未添加到训练数据集
TrainDoc=[]#训练文档集
DocTag=[]#训练文档的标签
jieba.add_word('不喜欢')
jieba.add_word('物联网')
def CallBack():
    return
def OpenDoc():
    global words
    filename = file.askopenfilename(title='打开文件名字', initialdir="Bayes",
                                    filetypes=[('docx文件', '*.docx')])
    doc1=docx.Document(filename)
    fullText=[]
    for para in doc1.paragraphs:
        print(para.text)
        fullText.append(para.text)
    print("len",len(fullText))
    ft=np.array(fullText)
    cft=''
    bd = ',!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+。，！？“”《》：、． '
    for i in range(len(ft)):
         cft=cft+ft[i]
    for i in bd:
        cft = cft.replace(i,'')
    words= jieba.lcut(cft)
    std.delete(0.0,tk.END)
    std.insert(tk.END, words)
    return

#添加一个训练文档
def AddDoc():
    global words,TagFlag,TrainDoc,DocTag,words,TargetVector,CountStr
    if (len(words) == 0 ):
        return
    if(TagFlag==False):
        mes.askokcancel(title="警告",message="没有选择类别")
        return
    TrainDoc.append(words)
    DocTag.append(TargetVector)
    CountStr.set(str(len(TrainDoc)))#显示当前训练文档计数
    TagFlag = False
    return
#保存训练文档样本
def SaveDoc():
    global TrainDoc,DocTag
    if (len(TrainDoc)== 0):
        return
    file1 = file.asksaveasfilename()
    if (file1 == ""):
        return
    np.savez(file1+".npz", Doc=TrainDoc,Tag=DocTag)
#装入入训练文档样本
def LoadDoc( ):
    global TrainDoc, DocTag
    file1 = file.askopenfilename(title='打开文件名字', initialdir="\Bayes",
                                 filetypes=[('npz文件', '*.npz')])
    if(file1==""):
        return
    ls=np.load(file1,allow_pickle=True)
    TrainDoc=list(ls['Doc'])
    DocTag=list(ls['Tag'])
    CountStr.set(str(len(TrainDoc)))
    std.delete(0.0, tk.END)
    std.insert(tk.END,TrainDoc)
#贝叶斯分类器学习训练
def BayesLearn(TrainDoc,DocTag ):
    #global TrainDoc, DocTag, TrainFlag,ClassNum
    DocNum=len(TrainDoc)#文档总数T
    #ClassNum =3#len(DocTag[0])#文档类别数
    cnt=np.zeros(ClassNum)#统计每类文档的个数，初值为0
    pv=np.zeros(ClassNum)#每类文档的先验概率，初值为0
    lt=[]
    m=len(TrainDoc)#文档计数
    #建立训练文档的词库wordlib
    for i in range(DocNum):
        for j in range(len(TrainDoc[i])):
            lt.append(TrainDoc[i][j])
    wordlib=np.unique(lt)#建立词库，消除重复单词
    WordNum=len(wordlib)#词库单词数|wordlib|
    #计算同类文档的先验概率pv
    TextClass=[[] for row in range(ClassNum)]#每个分类文档的联合列表的初始状态
    for i in range(DocNum):
        for j in range(ClassNum):
            if DocTag[i][j]==1.0:#训练文档的类别标签
                TextClass[j].append(TrainDoc[i])#将同类文档连接起来，注意这是一个三维list
                cnt[j]=cnt[j]+1#统计同类文档个数
    #计算类别的先验概率pv=P(vj)
    for i in range(ClassNum):
        pv[i]=cnt[i]/(cnt.sum())#相当讲义的P(vj)
    print(pv)
    #计算wk在每一类中出现的条件概率pk=P(wk|vj)
    pk=np.zeros((WordNum,ClassNum))#相当讲义的P(wk|vj)
    for i in range(WordNum):#遍历词库wordlib
         for j in range(ClassNum):#按类别循环
             nk=0#在某类文档中单词出现的次数
             nn=0#某类文档文档单词总数
             for k1 in range(len(TextClass[j])):#遍历第j类文档
                 nn=nn+len(TextClass[j][k1]) # 统计j类文档中单词总计数
                 for k2 in range (len(TextClass[j][k1])):#遍历j类文档中某一文档中的所有词汇
                     if wordlib[i]==TextClass[j][k1][k2]:
                           nk=nk+1# 在J类文档中单词wk 出现的次数
             pk[i][j]=(nk+1.0)/(nn+WordNum)
    return pv,pk,wordlib
#贝叶斯分类器
def BayesClassify(Doc,pv,pk,wordlib):
    ClassNum=len(pv)
    WordNum=np.size(pk,axis=0)
    p=np.zeros(ClassNum)
    for i in range(len(Doc)):#按position位置(属性）循环
        for j in range(WordNum):#寻找单词在词汇表中的位置
            if Doc[i]==wordlib[j]:
               for k in range(ClassNum):#计算所有本单词属于的类别条件概率
                   p[k]=p[k]+np.log(pk[j][k])
                   #while p[k]<0.000001:#防止连乘结果过小比例放大pk
                   #    p=p*10.0
    for i in range(ClassNum):
        p[i]=np.log(pv[i])+p[i]
    print(p)
    ClassIndex = np.argmax(p,axis=0)
    return ClassIndex,p

def CallBayesLearn():
    global TrainDoc, DocTag, TrainFlag
    if (len(TrainDoc)==0):
        return
    global pv,pk,wordlib
    pv,pk,wordlib=BayesLearn(TrainDoc, DocTag)
    np.savez("pvpk.npz", pv=pv, pk=pk, wordlib=wordlib)  # 将先验概率、条件概率和词库一起暂时存盘
    TrainFlag = True  # 训练完成标志
    EpochStr.set("训练结束")
    #std.delete(0.0, tk.END)
    #std.insert(tk.END, TrainDoc)
    print(pk)
    return
def SaveTrainResult():
    global pv,pk,wordlib
    if TrainFlag==False:
        return
    file1 = file.asksaveasfilename()
    if (file1 == ""):
        return
    np.savez(file1+".npz", pv=pv, pk=pk, wordlib=wordlib)  # 将先验概率、条件概率和词库一起存起个名字存盘
    return
def LoadTrainResult():
    global pv, pk, wordlib,TrainFlag
    file1 = file.askopenfilename(title='打开文件名字', initialdir="\pyproject\Bayes",
                                 filetypes=[('npz文件', '*.npz')])
    if (file1 == ""):
        TrainFlag = False
        return
    ls = np.load(file1, allow_pTickle=rue)
    pv = ls['pv']
    pk = ls['pk']
    wordlib = ls['wordlib']
    print(pk)
    TrainFlag=True
    return
def CallBayesClassify( ):
    global TrainFlag,words,DigitStr,pv,pk,wordlib
    #print(pk)
    if(TrainFlag==False):
        mes.askokcancel(title="警告", message="没有训练")
        return
    if(len(words)==0):
        mes.askokcancel(title="警告", message="没有文档")
        return
    #ls = np.load("pvpk.npz", allow_pickle=True)
    #pv = ls['pv']
    #pk = ls['pk']
    #wordlib=ls['wordlib']
    class_index,p=BayesClassify(words,pv,pk,wordlib)
    print(class_index)
    print(p)
    Result = ['喜欢', '一般','不喜欢']
    ResultStr.set(Result[class_index])
    MainWin.mainloop()
    return
ww = 500
wh = 400
def picshow(filename):
    I1 = cv2.imread(filename)
    I2 = cv2.resize(I1, (ww, wh))
    cv2.imwrite('image/temp.png', I2)
    I3 = tk.PhotoImage(file='image/temp.png')
    L1 = tk.Label(InputFrame, image=I3)
    L1.grid(row=1, column=0, columnspan=2, padx=40)
    MainWin.mainloop()
    return


def SetTarget():
    global TagFlag, TargetVector
    TagFlag = True
    for i in range(10):
        if (TargetV.get() == i):
            TargetVector = np.zeros(10)
            TargetVector[i] = 1.0
            print(TargetVector)


# =========菜单区=============
menubar = tk.Menu(MainWin)
# file menu
fmenu = tk.Menu(menubar)
fmenu.add_command(label='新建', command=CallBack)
fmenu.add_command(label='打开', command=CallBack)
fmenu.add_command(label='保存', command=CallBack)
fmenu.add_command(label='另存为', command=CallBack)
# Image processing menu
imenu = tk.Menu(menubar)
imenu.add_command(label='FFT变换', command=CallBack)
imenu.add_command(label='DOT变换', command=CallBack)
imenu.add_command(label='边缘检测', command=CallBack)
imenu.add_command(label='区域分割', command=CallBack)
# machine learning
mmenu = tk.Menu(menubar)
mmenu.add_command(label='KNN', command=CallBack)
mmenu.add_command(label='朴素贝叶斯', command=CallBack)
mmenu.add_command(label='支持向量机', command=CallBack)
mmenu.add_command(label='BP神经网', command=CallBack)
mmenu.add_command(label='CNN卷积神经网', command=CallBack)
# =============
menubar.add_cascade(label="文件操作", menu=fmenu)
menubar.add_cascade(label="图像处理", menu=imenu)
menubar.add_cascade(label="机器学习", menu=mmenu)
MainWin.config(menu=menubar)
# 设置4个Frame 区，
InputFrame = tk.Frame(MainWin, height=300, width=600, relief=tk.RAISED)
OutputFrame = tk.Frame(MainWin, height=300, width=600, relief=tk.RAISED)
ButFrame = tk.Frame(MainWin, height=300, width=600)
DataFrame = tk.Frame(MainWin, height=300, width=600)
InputFrame.grid(row=0, column=0)
OutputFrame.grid(row=0, column=1)
ButFrame.grid(row=1, column=0)
DataFrame.grid(row=1, column=1, sticky=tk.N)
#---InputFrame---输入框架
ResultStr = tk.StringVar()
Lab1 = tk.Label(InputFrame, text='识别结果:', font=('Arial', 12), width=20, height=1)
Lab1.grid(row=0, column=0, padx=10, pady=20)
entry1 = tk.Entry(InputFrame, font=('Arial', 12), width=20, textvariable=ResultStr)
entry1.grid(row=0, column=1)
ResultStr.set('结果字符串')
LogoImage = tk.PhotoImage(file='image/Bayeslogo.png')
Lab2 = tk.Label(InputFrame, image=LogoImage)
Lab2.grid(row=1, column=0, columnspan=2, padx=20)
#---OutputFrame---输出框架
Lab3 = tk.Label(OutputFrame, text='训练文档', font=('Arial', 14), width=10, height=1)
Lab3.grid(row=0, column=0, pady=20)
std=st.ScrolledText(OutputFrame,width=30,height=14,font=("宋体",16))
std.grid(row=1, column=0,padx=40)
#---Button Frame---按钮框架
Target = [('喜欢', 0), ('一般', 1), ('讨厌', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9)]
TargetVector = np.zeros(10)
TargetV = tk.IntVar()
Target_startx = 10
Target_starty = 10
for txt, num in Target:
    rbut = tk.Radiobutton(ButFrame, text=txt, value=num, font=('Arial', 12), width=3, height=1, command=SetTarget,
                          variable=TargetV)
    rbut.place(x=Target_startx + num * 70, y=Target_starty)
but1=tk.Button(ButFrame, text='打开文档', font=('Arial', 12), width=10, height=1, command=OpenDoc)
but2=tk.Button(ButFrame, text='添加样本', font=('Arial', 12), width=10, height=1, command=AddDoc)
but3=tk.Button(ButFrame, text='保存样本', font=('Arial', 12), width=10, height=1, command=SaveDoc)
but4=tk.Button(ButFrame, text='装入样本', font=('Arial', 12), width=10, height=1, command=LoadDoc)
but5=tk.Button(ButFrame, text='Bayes学习', font=('Arial', 12), width=10, height=1, command=CallBayesLearn)
but6=tk.Button(ButFrame, text='Bayes分类', font=('Arial', 12), width=10, height=1, command=CallBayesClassify)
but7=tk.Button(ButFrame, text='保存训练结果', font=('Arial', 12), width=10, height=1, command=SaveTrainResult)
but8=tk.Button(ButFrame, text='装入训练结果', font=('Arial', 12), width=10, height=1, command=LoadTrainResult)
but9=tk.Button(ButFrame, text='备用按钮', font=('Arial', 12), width=10, height=1, command=CallBack)
dd=40
but1.place(x=50, y=40)
but2.place(x=250, y=40)
but3.place(x=450, y=40)
but4.place(x=50, y=80)
but5.place(x=250, y=80)
but6.place(x=450, y=80)
but7.place(x=50, y=120)
but8.place(x=250, y=120)
but9.place(x=450, y=120)
#--- Data Frame---数据显示框架
Lab4 = tk.Label(DataFrame, text='样本计数:', font=('Arial', 12), width=10, height=1)
Lab4.grid(row=0, column=0, pady=20)
Lab5 = tk.Label(DataFrame, text='训练次数:', font=('Arial', 12), width=10, height=1)
Lab5.grid(row=1, column=0)
Lab6 = tk.Label(DataFrame, text='训练误差:', font=('Arial', 12), width=10, height=1)
Lab6.grid(row=2, column=0, pady=20)
#---显示样本计数----
CountStr = tk.StringVar()
entry4 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=CountStr)
entry4.grid(row=0, column=1, pady=20)
CountStr.set("0")
#---显示训练次数---
EpochStr = tk.StringVar()
entry5 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=EpochStr)
entry5.grid(row=1, column=1)
EpochStr.set("0")
#---显示训练误差---
ErrStr = tk.StringVar()
entry6 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=ErrStr)
entry6.grid(row=2, column=1, pady=20)
ErrStr.set("0")
MainWin.mainloop()
