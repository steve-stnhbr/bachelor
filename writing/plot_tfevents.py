import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import click

@click.command()
@click.argument("input_file")
def main(input_file):
    if "," in input_file:
        print("Multiple files not supported yet!")
        input_files = input_file.split(",")

    def extract_metrics_from_tfevents(filepath):
        metrics = {}
        for event in summary_iterator(filepath):
            for value in event.summary.value:
                if value.tag not in metrics:
                    metrics[value.tag] = []
                metrics[value.tag].append((event.wall_time, event.step, value.simple_value))
        return metrics

    # Replace with your tfevents file path
    tfevents_file = input_file

    extracted_metrics = extract_metrics_from_tfevents(tfevents_file)

    # Print the extracted metrics
    for tag, values in extracted_metrics.items():
        print(f"Metric: {tag}")
        for time, step, value in values[:5]:  # Print first 5 entries
            print(f"  Time: {time}, Step: {step}, Value: {value}")
        print(f"  ... ({len(values)} entries total)")


if __name__ == '__main__':
    main()