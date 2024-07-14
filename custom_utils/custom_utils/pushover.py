import os
import click
import http.client, urllib
from pushover import Pushover

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
    from pushover import Pushover

    po = Pushover(API_TOKEN)
    po.user(USER_KEY)
    msg = po.msg(message)
    msg.set("title", title)
    po.send(msg)

@click.command()
@click.argument('message')
@click.option('-t', '--title', type=str, default=None)
def main(message, title):
    send_pushover_notification(message, title)

if __name__ == '__main__':
    main()