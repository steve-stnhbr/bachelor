import os
import click
from pushover import Client

USER_KEY = os.environ["PUSHOVER_USER_KEY"]
API_TOKEN = os.environ["PUSHOVER_API_TOKEN"]

def send_pushover_notification(message, title=None):
    client = Client(USER_KEY, api_token=API_TOKEN)
    client.send_message(message, title=title)

@click.command()
@click.argument('message')
@click.option('-t', '--title', type=str, default=None)
def main(message, title):
    send_pushover_notification(message, title)
