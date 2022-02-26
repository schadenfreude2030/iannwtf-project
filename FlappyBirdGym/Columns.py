import random

class Columns:
    id = 0
    def __init__(self, canvas, posX, maxHeight, column_width, previousTopHeight):
        
        self.canvas = canvas
        self.posX = posX

        self.id = Columns.id

        self.column_width = column_width

        self.birdFlownOver = False

        self.top_height = random.randint(previousTopHeight - 30, previousTopHeight + 30)
        self.free_space = random.randint(60, 80)


        self.middle_point = self.top_height + self.free_space/2

        self.tag_column_top = f"square_top_{self.id}"
        self.column_top = self.canvas.create_rectangle(
                            posX, 0, posX + self.column_width, self.top_height,
                            fill='white',
                            outline='white',
                            tags=(self.tag_column_top)
                            )

        self.tag_column_down = f"square_down_{self.id}"
        self.column_down = self.canvas.create_rectangle(
                            posX, self.top_height + self.free_space, posX + self.column_width, maxHeight,
                            fill='white',
                            outline='white',
                            tags=(self.tag_column_down)
                            )

        Columns.id += 1
    
    def getPosX(self):
        return self.posX

    def move(self, dx, dy):
        
        self.posX += dx

        self.canvas.move(self.tag_column_top, dx, dy)
        self.canvas.move(self.tag_column_down, dx, dy)
    

    def delete(self):
        
        self.canvas.delete(self.column_top)
        self.canvas.delete(self.column_down)

    def touched(self):

        def touchedSingleColumn(pos_bird, pos_column):
            bird_x1, bird_y1, bird_x2, bird_y2 = pos_bird
            square_x1, square_y1, square_x2, square_y2 = pos_column

            left_side = ( (square_y1 <= bird_y1 and bird_y1 <= square_y2) or 
                          (square_y1 <= bird_y2 and bird_y2 <= square_y2) ) and square_x1 <= bird_x2 and bird_x2 <= square_x2

            top_side =  ( (square_x1 <= bird_x1 and bird_x1 <= square_x2) or 
                          (square_x1 <= bird_x2 and bird_x2 <= square_x2) ) and square_y1 <= bird_y2 and bird_y2 <= square_y2

            right_side = ( (square_y1 <= bird_y1 and bird_y1 <= square_y2) or 
                           (square_y1 <= bird_y2 and bird_y2 <= square_y2) ) and bird_x1 <= square_x2 and square_x1 <= bird_x1

            down_side = ( (square_x1 <= bird_x1 and bird_x1 <= square_x2) or 
                          (square_x1 <= bird_x2 and bird_x2 <= square_x2) ) and bird_y1 <= square_y2 and square_y1 <= bird_y1
    
            return left_side or top_side or right_side or down_side


        pos_bird = self.canvas.coords('bird')
        pos_column_top = self.canvas.coords(self.tag_column_top)
        pos_column_down = self.canvas.coords(self.tag_column_down)

        return touchedSingleColumn(pos_bird, pos_column_top) or touchedSingleColumn(pos_bird, pos_column_down)

    def getTopHeight(self):
        return self.top_height
    
    def wasBirdFlownOver(self):
        return self.birdFlownOver
    
    def setBirdFlownOver(self, birdFlownOver):
        self.birdFlownOver = birdFlownOver
        