def yolobbox2bbox(x,y,w,h, height, width):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1*width, y1*height, x2*width, y2*height