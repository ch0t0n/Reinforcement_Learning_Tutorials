from turtle import Screen, Turtle

GRID_SIZE = 600

sub_divisions = 10

cell_size = GRID_SIZE / float(sub_divisions)  # force float for Python 2

screen = Screen()

turtle = Turtle()
turtle.hideturtle()
turtle.speed(0)
turtle.penup()
turtle.goto(-GRID_SIZE/2, GRID_SIZE/2)
turtle.pendown()

angle = 90

for _ in range(4):
    turtle.forward(GRID_SIZE)
    turtle.right(angle)

for _ in range(2):
    for _ in range(1, sub_divisions):
        turtle.forward(cell_size)
        turtle.right(angle)
        turtle.forward(GRID_SIZE)
        turtle.left(angle)

        angle = -angle

    turtle.forward(cell_size)
    turtle.right(angle)

turtle2 = Turtle()
turtle2.shape("turtle")
turtle2.hideturtle()
turtle2.speed(0)
turtle2.penup()
turtle2.goto(GRID_SIZE/(10*2), GRID_SIZE/(10*2))
turtle2.pendown()
turtle2.showturtle()

screen.exitonclick()