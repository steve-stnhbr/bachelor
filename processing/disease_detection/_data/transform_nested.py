import os
import shutil
import click

@click.command()
@click.option('-i', '--input')
@click.option('-o', '--output')
def main(input, output):
    '''
    This script expects a dataset that has the following structure

    - data
      - plant_type
        - healthy
        - diseased
        - <other_classes>
    
    and will transform it into the following strucure

    - data
      - plant_type__<class>
      - plant_type__healthy
      - plant_type__diseased
    '''

    for plant in os.listdir(input):
        for clazz in os.listdir(os.path.join(input, plant)):
            os.makedirs(os.path.join(output, f"{plant}__{clazz}"), exist_ok=True)
            for file in os.listdir(os.path.join(input, plant, clazz)):
                shutil.copy(os.path.join(input, plant, clazz, file), os.path.join(output, f"{plant}__{clazz}", file))

if __name__ == '__main__':
    main()