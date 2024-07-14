import os
import click
import http.client, urllib
from pushover import Client

def send_pushover_notification(message, title=None):
    USER_KEY = os.getenv("PUSHOVER_USER_KEY")
    API_TOKEN = os.getenv("PUSHOVER_API_TOKEN")
    if "PUSHOVER_USER_KEY" is None:
        print("No user key provided, aborting pushover notification")
        print("Consider adding 'PUSHOVER_USER_KEY' to your environment variables")
        return
    if "PUSHOVER_API_TOKEN" is None:
        print("No user key provided, aborting pushover notification")
        print("Consider adding 'PUSHOVER_API_TOKEN' to your environment variables")
        return
    client = Client(USER_KEY, api_token=API_TOKEN)
    client.send_message(message, title=title)
    print("Sent notification")

@click.command()
@click.argument('message')
@click.option('-t', '--title', type=str, default=None)
def main(message, title):
    send_pushover_notification(message, title)

if __name__ == '__main__':
    main()