from tkinter import *
from tkinter.filedialog import askopenfilename
from main import recognition

def select():
    file_path = askopenfilename(title=u'选择文件', filetypes=[(".JPG", ".jpg")])
    number.set(recognition(file_path))
    

root = Tk()
root.title('车牌检测')
number = StringVar()
Label(root, text= '车牌号').grid(row = 0, column = 0)
Entry(root, textvariable = number).grid(row = 0, column = 1)
Button(root, text = '打开图片', command = select).grid(row = 1, column = 0)
Button(root, text = '退出', command = root.quit).grid(row = 1, column = 3)
root.mainloop()
