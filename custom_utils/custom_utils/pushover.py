import os
import click
import http.client, urllib


def send_pushover_notification(message, title=None):
    if "PUSHOVER_USER_KEY" not in os.environ:
        print("No user key provided, aborting pushover notification")
        print("Consider adding 'PUSHOVER_USER_KEY' to your environment variables")
        return
    if "PUSHOVER_API_TOKEN" not in os.environ:
        print("No user key provided, aborting pushover notification")
        print("Consider adding 'PUSHOVER_API_TOKEN' to your environment variables")
        return
    USER_KEY = os.environ["PUSHOVER_USER_KEY"]
    API_TOKEN = os.environ["PUSHOVER_API_TOKEN"]
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
    urllib.parse.urlencode({
        "token": API_TOKEN,
        "user": USER_KEY,
        "message": message,
        "title": title
    }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()

@click.command()
@click.argument('message')
@click.option('-t', '--title', type=str, default=None)
def main(message, title):
    send_pushover_notification(message, title)
