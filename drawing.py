import tkinter as tk

# import CNN

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing App")

        self.canvas_width = 500
        self.canvas_height = 500

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.setup_buttons()
        self.setup_events()

    def setup_buttons(self):
        self.save_button = tk.Button(self.master, text="Finish", command=self.generate_new)
        self.save_button.pack()

    def setup_events(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.master.bind("<KeyPress-s>", self.generate_new)

    def draw(self, event):
        x, y = event.x, event.y
        r = 20
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#000000", outline="")
            
    def generate_new(self, event=None):
        mask = self.generate_mask()

        

def main():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()