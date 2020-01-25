import Project as pj

def result():
    house_age = float(e1.get())
    dist = float(e2.get())
    stores = float(e3.get())
    lat = float(e4.get())
    lon = float(e5.get())
    data = np.array([house_age, dist, stores, lat, lon])
    data = (data - data.mean()) / data.std()
    result = pj.p.predict(data)
    e6.delete(0, 100)
    e6.insert(0, round(result, 2))
    

from tkinter import *
m = Tk()
m.title("Real Estate Valuation")

l1 = Label(m, text = "House Age").grid(row = 0)
l2 = Label(m, text = "Distance to the nearest MRT station").grid(row = 1)
l3 = Label(m, text = "Number of convenience stores").grid(row = 2)
l4 = Label(m, text = "Latitude").grid(row = 3)
l5 = Label(m, text = "Longitude").grid(row = 4)
e1 = Entry(m, bd = 4)
e2 = Entry(m, bd = 4)
e3 = Entry(m, bd = 4)
e4 = Entry(m, bd = 4)
e5 = Entry(m, bd = 4)
e1.grid(row = 0, column = 1, padx = 3, pady = 3)
e2.grid(row = 1, column = 1, padx = 3, pady = 3)
e3.grid(row = 2, column = 1, padx = 3, pady = 3)
e4.grid(row = 3, column = 1, padx = 3, pady = 3)
e5.grid(row = 4, column = 1, padx = 3, pady = 3)
b1 = Button(m, text = "Predict", width = 26, command = result)
b1.grid()
e6 = Entry(m, bd = 5)
e6.grid(row = 5, column = 1, padx = 15, pady = 15)
m.mainloop()
