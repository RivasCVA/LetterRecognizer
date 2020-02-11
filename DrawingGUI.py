
from tkinter import *
from tkinter import ttk


class main:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)  # drawing the line
        self.c.bind('<ButtonRelease-1>', self.reset) #when releasing the left mouse button, it will stop drawing.

    def paint(self, e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, fill=self.color_fg,
                               capstyle=ROUND, smooth=True)
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):  # reseting or cleaning the canvas
        self.old_x = None
        self.old_y = None

    def clear(self):
       self.c.delete(ALL)
       
    def drawWidgets(self): #setting, need save/engine documentation
        self.c = Canvas(self.master, width=148, height=148)
        self.c.pack(fill=BOTH, expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        
        optionmenu = Menu(menu)
        
        menu.add_cascade(label='Options', menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas', command=self.clear)
        optionmenu.add_command(label='Exit', command=self.master.destroy)


if __name__ == '__main__':
    root = Tk()
    root.resizable(False, False) #this doesn't allot resize
    main(root)
    root.title('Drawing Pad')
    root.mainloop()
    
