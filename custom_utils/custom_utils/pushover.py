import os
import click
import http.client, urllib

USER_KEY = os.environ["PUSHOVER_USER_KEY"]
API_TOKEN = os.environ["PUSHOVER_API_TOKEN"]

def send_pushover_notification(message, title=None):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
    urllib.parse.urlencode({
        "token": APP_TOKEN,
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
