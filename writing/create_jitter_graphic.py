import pixie
import click
import random
import cv2

IMAGE_SIZE = (512, 512)

def draw_division_line(position, axis, color, width):
    if axis == 'x':
        x0 = 0
        x1 = IMAGE_SIZE[0]
        y0 = position
        y1 = position
    elif axis == 'y':
        y0 = 0
        y1 = IMAGE_SIZE[1]
        x0 = position
        x1 = position
    else:
        raise Error("Axis not in ('x', 'y')")

    draw_line((x0, y0), (x1, y1), color, width)

def draw_line(start, end, color, width):
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = pixie.Color(color[0], color[1], color[2], 1)
    global image
    ctx = image.new_context()
    ctx.stroke_style = paint
    ctx.line_width = width

    ctx.stroke_segment(start[0], start[1], end[0], end[1])

def draw_dot(position, radius, color):
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = pixie.Color(color[0], color[1], color[2], 1)
    global image
    ctx = image.new_context()
    ctx.fill_style = paint
    ctx.rounded_rect(position[0] - radius, position[1] - radius, radius * 2, radius * 2, radius, radius, radius, radius)
    ctx.fill()

@click.command()
@click.option("-s", "--subdivisions", type=int)
def main(subdivisions):
    global image
    image = pixie.Image(IMAGE_SIZE[0], IMAGE_SIZE[1])
    image.fill(pixie.Color(1, 1, 1, 1))
    width = IMAGE_SIZE[0] / subdivisions
    height = IMAGE_SIZE[1] / subdivisions
    for i in range(subdivisions):
        draw_division_line(i * width, 'x', (195, 195, 195), 3)
        draw_division_line(i * height, 'y', (195, 195, 195), 3)
        for j in range(subdivisions):
            x_min = int(i * width)
            x_max = int((i + 1) * width - 1)
            y_min = int(j * height)
            y_max = int((j + 1) * height - 1)

            x = random.randrange(x_min, x_max)
            y = random.randrange(y_min, y_max)

            draw_dot((x, y), 4, (210, 95, 22))
    image.write_file("output.png")


if __name__ == '__main__':
    main()