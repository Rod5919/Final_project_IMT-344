import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from overlap import *

rect = []
overlaped = 0
# True
rect.append(rectangle(0,3,4,-1))
rect.append(rectangle(2,2,5,0))
rect.append(rectangle(2,0,5,-2))
rect.append(rectangle(-2,0,1,-2))
rect.append(rectangle(-2,-2,1,0))

# False
rect.append(rectangle(5,5,6,4))
rect.append(rectangle(5,-2,6,-4))
rect.append(rectangle(-2,-2,-1,-3))
rect.append(rectangle(-4,5,-1,4))

rect.append(rectangle(5,5,6,0))
rect.append(rectangle(5,1,6,-4))
rect.append(rectangle(-2,0,-1,-2))
rect.append(rectangle(-4,5,-1,2))

plot_rect = tuple(rect)

for i in range(1,len(rect)):
    for j in range(1,len(rect)):
        if overlap(rect[0],rect[j]):
            overlaped +=1
            rect[0].coordinates()
            rect[j].coordinates()
            print()
            break
    rect.pop(0)    

print(overlaped)

def rect2list(r):
    return [r.x1, r.y1, r.x2, r.y2]

l = [rect2list(x) for x in plot_rect]
flat_l = [x for sublist in l for x in sublist]

X = flat_l[::2]
Y = flat_l[1::2]

for i in range(overlaped):
    plt.figure

    #define Matplotlib figure and axis
    fig, ax = plt.subplots()

    #create simple line plot
    ax.plot([0, 10],[0, 0])

    #add rectangle to plot
    ax.add_patch(Rectangle((X[i], Y[i+1]), X[i+1]-X[i], Y[i]-Y[i+1], edgecolor = 'red', fill = False))
    ax.add_patch(Rectangle((X[i+2], Y[i+1+2]), X[i+1+2]-X[i+2], Y[i+2]-Y[i+1+2], edgecolor = 'blue', fill = False))

    #display plot
    plt.show()