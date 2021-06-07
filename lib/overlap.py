class rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def coordinates(self):
        print(f"[{self.x1},{self.y1},{self.x2},{self.y2}]")

def overlap(r1, r2):
    return (r1.x1 < r2.x2 and r1.x2 > r2.x1 and
    ((r1.y1 > r2.y2 and r1.y2 < r2.y1) or (r1.y2 > r2.y1 and r1.y2 < r2.y2)))