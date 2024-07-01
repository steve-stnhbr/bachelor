import os
import click
import shutil

@click.command()
@click.option('-i', '--input', type=str)
@click.option('-o', '--output', type=str)
@click.option('-d', '--indicator', type=str)
@click.option('-n', '--negative', type=str, default=None)
def main(input, output, indicator, negative):
    if negative is None:
        negative = "not_" + indicator
    pos_out = os.path.join(output, indicator)
    neg_out = os.path.join(output, negative)
    os.makedirs(pos_out)
    os.makedirs(neg_out)
    for dir in os.listdir(input):
        if not os.path.isdir(dir):
            continue
        out = pos_out if indicator in dir else neg_out
        shutil.copy(os.path.join(input, dir), out)

if __name__ == '__main__':
    main()