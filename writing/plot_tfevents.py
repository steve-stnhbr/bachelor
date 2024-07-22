import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import click

@click.command()
@click.argument("input_file")
def main(input_file):
    if "," in input_file:
        print("Multiple files not supported yet!")
        input_files = input_file.split(",")

    for event in summary_iterator(input_file):
        print(event.step)

if __name__ == '__main__':
    main()