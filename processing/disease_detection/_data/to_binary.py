import os
import click
import shutil
import uuid

@click.command()
@click.option('-i', '--input', type=str)
@click.option('-o', '--output', type=str)
@click.option('-d', '--indicator', type=str)
@click.option('-n', '--negative', type=str, default=None)
@click.option('-c', '--case', is_flag=True)
def main(input, output, indicator, negative, case):
    if negative is None:
        negative = "not_" + indicator
    pos_out = os.path.join(output, indicator)
    neg_out = os.path.join(output, negative)
    os.makedirs(pos_out, exist_ok=True)
    os.makedirs(neg_out, exist_ok=True)
    for dir in os.listdir(input):
        if os.path.isfile(os.path.join(input, dir)):
            continue
        if case:
            out = pos_out if indicator.lower() in dir.lower() else neg_out
        else:
            out = pos_out if indicator in dir else neg_out
        for file in os.listdir(os.path.join(input, dir)):
            file_path = os.path.join(input, dir, file)
            if os.path.isdir(file_path):
                continue
            file_name = uuid.uuid4().hex + os.path.splitext(file)[1]
            shutil.copy(file_path, os.path.join(out, file_name))

if __name__ == '__main__':
    main()