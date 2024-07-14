import os
import click
import http.client, urllib
import requests

def send_pushover_notification(message, title=None):
    USER_KEY = os.getenv("PUSHOVER_USER_KEY")
    API_TOKEN = os.getenv("PUSHOVER_API_TOKEN")
    if "PUSHOVER_USER_KEY" == None:
        print("No user key provided, aborting pushover notification")
        print("Consider adding 'PUSHOVER_USER_KEY' to your environment variables")
        return
    if "PUSHOVER_API_TOKEN" == None:
        print("No user key provided, aborting pushover notification")
        print("Consider adding 'PUSHOVER_API_TOKEN' to your environment variables")
        return
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
            "token": "APP_TOKEN",
            "user": "USER_KEY",
            "message": "hello world"
        },
        files = {
            "attachment": ("image.jpg", open("your_image.jpg", "rb"), "image/jpeg")
        },
        timeout=800
    )

@click.command()
@click.argument('message')
@click.option('-t', '--title', type=str, default=None)
def main(message, title):
    send_pushover_notification(message, title)

if __name__ == '__main__':
    main()