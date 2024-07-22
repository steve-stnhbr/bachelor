import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import click
import pandas as pd
from pathlib import Path

@click.command()
@click.argument("input_file")
@click.option("-o", "--output", default="out")
def main(input_file, output):
    if "," in input_file:
        print("Multiple files not supported yet!")
        input_files = input_file.split(",")

    def extract_metrics_to_dataframe(filepath):
        data = []
        for event in summary_iterator(filepath):
            for value in event.summary.value:
                data.append({
                    'tag': value.tag,
                    'wall_time': event.wall_time,
                    'step': event.step,
                    'value': value.simple_value
                })
        return pd.DataFrame(data)

    df = extract_metrics_to_dataframe(input_file)

    # Display basic information about the DataFrame
    print(df.info())

    # Display the first few rows
    print(df.head())

    # Get unique metric names
    metric_names = df['tag'].unique()
    print("Metrics found:", metric_names)

    # Example: Get statistics for each metric
    for metric in metric_names:
        metric_data = df[df['tag'] == metric]
        print(f"\nMetric: {metric}")
        print(metric_data['value'].describe())
    
    df.to_csv(f"{output}/{Path(input_file).parent.stem}.csv")


if __name__ == '__main__':
    main()