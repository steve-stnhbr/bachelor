import os
import click
import http.client, urllib
import requests

def send_pushover_notification(message, title=None, file=None, file_format=None):
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
    data = {
            "token": API_TOKEN,
            "user": USER_KEY,
            "message": message,
            "title": title
        }
    files = None
    if file is not None:
        files = {
            "attachment": (file, open(file, "rb"), file_format)
        }
    r = requests.post("https://api.pushover.net/1/messages.json", data=data, files=files,timeout=800)
    print("Sent notification:", title)
    return r

@click.command()
@click.argument('message')
@click.option('-t', '--title', type=str, default=None)
def main(message, title):
    send_pushover_notification(message, title)

if __name__ == '__main__':
    main()